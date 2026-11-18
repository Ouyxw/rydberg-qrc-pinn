# 基于 Sagittarius 的 Rydberg-QRC-PINN 研究实施方案

> 版本：v0.1（2026-11-17）  
> 研究对象：欧式看涨期权 Black–Scholes PDE  
> 核心路线：解析基线 → MLP-PINN → 经典随机特征 PINN → 固定 Rydberg-QRC-PINN → 20 原子规模扩展  
> 代码基础：[Sagittarius](https://github.com/Qosmos-Technology/Sagittarius)（Python SDK + Julia 求解后端）  
> 状态：**研究计划，不构成已实现的性能声明**。

## 0. 执行摘要

**科学问题：**在相同金融 PDE 定价误差、Greeks 误差和总计算资源约束下，Rydberg 多体动力学产生的固定量子特征，能否比经典 MLP 或同等规模的经典随机特征更有效地逼近金融 PDE 的解？

**核心原则：**

1. 固定量子储备池提供特征映射，不负责训练全部经典 PINN 权重；只优化经典读出权重。
2. 在 Black–Scholes 的线性 PDE 和线性读出假设下，训练可写成带正则项的最小二乘问题；因此要比较**函数逼近与全流程耗时**，而非只比较优化迭代次数。
3. 必须设置三组基线：解析解 / FDM 数值基线；MLP-PINN；经典随机特征 PINN（如 RBF / Random Fourier Features）。
4. 首期不把噪声硬件或量子优势当作既成事实。20 原子以本机实际运行数据验收。
5. 优先解决输入编码和 PDE 一、二阶导数估计，再提升原子数。

### 0.1 设备约束

| 部件 | 设备及资源 | 项目使用方式 |
|---|---|---|
| CPU | Intel Core i7-14700KF，20 核 | 经典基线、仿真 CPU 回退、数据处理 |
| RAM | 32 GB | Julia / Python 工作集、样本特征缓存 |
| GPU | GeForce RTX 5070 Ti，16 GB | 经典 PINN、CUDA Rydberg 模拟（须验证） |
| SSD | 1 TB | 项目环境、模型和基准记录 |
| 系统建议 | Windows 11 + WSL2 Ubuntu / Ubuntu | NVIDIA 驱动、CUDA、Julia、Python 工具链 |

对于二能级原子的**单个完整纯态向量**，20 原子需要 $2^{20}$ 个复振幅；`complex128` 仅状态向量占 **16 MiB**，但实际积分器、算符及中间数组可能消耗更多内存。不能把状态存储量当作仿真总显存；也不能使用维度 $2^{20}\times2^{20}$ 的稠密哈密顿量。GPU CUDA 后端在公开 Sagittarius 文档中标为**实验性**，需要先做运行及精度确认。

### 0.2 仓库现状与新增能力

当前公开 Sagittarius README 记载支持二维/三维阵列、完整及 blockade 缩减 Hilbert 空间、全局/局域 Rabi 和失谐控制、Schrödinger/Lindblad/MCWF、单体布居、参数扫描、CPU 以及实验性 CUDA；求解器为 `Tsit5`、`Vern9`、`RK4`。此处不是对全部接口实现或 20 原子性能的独立审计。

| 能力 | 现有基础 | 本项目处理方式 |
|---|---|---|
| Rydberg 物理模型与时序 | 有 | 直接依赖 Sagittarius |
| 原子坐标 / blockade reduced basis | 有 | 基准中分别比较 full 与 reduced |
| 单原子布居 | 有 | 初版特征 $\langle n_i\rangle$ |
| 双体关联 $\langle n_i n_j\rangle$ | 公开顶层接口尚未核实 | 先经小规模状态向量/bitstring 验证，再实现独立适配层 |
| 输入 $(s,\tau)$ 到脉冲编码 | 有局域控制基础，未有专门 QRC 契约 | 实现 `InputEncoder` |
| $\partial_s\phi,\partial_{ss}\phi,\partial_\tau\phi$ | 未核实统一接口 | 首期有限差分；后期灵敏度 ODE / AD |
| 批量批处理 + 缓存 | 尚待核实 | 独立实验项目实现，不破坏 Sagittarius 原公共 API |

## 1. 数学任务和数据规格

### 1.1 欧式看涨期权

设 $\tau=T-t$ 是金融合约**剩余期限**（不是量子演化时间），Black–Scholes PDE 为

$$
\frac{\partial V}{\partial\tau}-\frac12\sigma^2S^2\frac{\partial^2V}{\partial S^2}-rS\frac{\partial V}{\partial S}+rV=0.
$$

到期 payoff 与计算域边界：

$$
V(S,0)=\max(S-K,0),\quad V(0,\tau)=0,\quad V(S_{\max},\tau)\approx S_{\max}-Ke^{-r\tau}.
$$

首期固定 $K=50$、$r=0.05$、$\sigma=0.2$、$T=1$、$S\in[0,150]$，与 DeepXDE 官方示例对齐。选择 $S_{\max}=3K$ 时上边界为**截断近似条件**，在独立解析解评估中记录截断误差。

无量纲化：$s=S/K$、$u=V/K$、$x=(s,\tau)$，得到

$$
\boxed{\mathcal L u=\partial_\tau u-\tfrac12\sigma^2s^2\partial_{ss}u-rs\partial_su+ru=0},
$$

$$u(s,0)=\max(s-1,0),\qquad s\in[0,3],\quad\tau\in[0,1].$$

**验证真值：**Black–Scholes 解析价格及其解析 $\Delta,\Gamma$；价格接近零的区域避免逐点相对百分比误差失真。定义 $\partial_S V=\partial_su$、$\partial^2_S V=K^{-1}\partial^2_su$，以保证 Greeks 单位正确。

### 1.2 数据划分

- 固定评估网格：例如 $256\times128$ 点；不参与训练，所有模型统一使用。
- 初始域内 collocation 集：经典基线从 2,048 或 4,096 点开始；采用 Sobol 采样，抽样数可取 $2^k$；边界和初始点另选。
- 初始 quantum collocation：16/32/64/128 点阶梯试验；不要从 20 原子、数千点直接启动。
- 验证区域：全域；另在 $\tau\ge0.05$、$s\in[0.5,2.5]$ 的平滑区域独立评估 Gamma；重点加密 $s\approx1,\tau\approx0$ 的 payoff kink。
- 随机性：训练种子至少 5 个；20 原子高成本实验如资源不足，报告每个试验的重复次数和置信区间，不虚构稳定性结论。

### 1.3 金融数据集的分层定义与角色

**研究原则：PDE collocation points 不是市场报价数据。** M0–M4 的首要目的，是比较同一个已知 Black–Scholes 方程下不同函数表示的数值能力；首期无须购买期权历史数据。为避免把模型误设定（例如固定波动率无法拟合波动率微笑）误判为 QRC 特征失效，建议划分以下数据层级：

| 层级 | 内容 | 是否带价格标签 | 进入训练目标 | 应用阶段 |
|---|---|---|---|---|
| **D0 物理约束训练集** | $x=(s,\tau)$ 内部配点、到期条件、边界点；PDE 系数 $r,\sigma$ | 内部配点**无价格标签**；边界有解析约束目标 | PDE residual、初始/边界损失 | M0–M4 必选 |
| **D1 解析参考集** | Black–Scholes 解析价格、Delta、Gamma 及独立测试网格 | **有**，由解析公式计算 | 默认不用于训练；独立验证，可另设监督消融 | M0–M4 必选 |
| **D2 参数化合成集** | 不同 $K,r,\sigma,T$ 与状态变量组合生成的价格/Greeks | **有**，由解析模型生成 | 用于泛化测试或参数化网络训练，须明确实验设定 | M3–M5 可选 |
| **D3 真实期权市场集** | 欧式期权 Bid/Ask、标的价格、到期日、利率、远期/股息信息 | **有噪声**，且不是 Black–Scholes 真值 | 单独的市场数据拟合/校准试验 | M5 扩展，不作为 M0–M4 的前提 |

#### 1.3.1 D0：PINN 无监督 PDE 训练样本

- **内部配点**：首期在 $s\in[0,3],\tau\in(0,1]$ 内使用 Sobol 序列生成 $2^{10}$、$2^{11}$、$2^{12}$ 个候选点，采用**固定且与测试集不同的随机种子/扰乱序列**。量子初期从其中固定抽取 16/32/64/128 个点以控制昂贵的演化次数。
- **初始条件点**：$\tau=0$，目标 $u(s,0)=\max(s-1,0)$；在 $s\simeq1$ 附近适当加密，但不能把 kink 邻域的导数误差视作常规平滑误差。
- **边界条件点**：$s=0$ 和 $s=3$，目标分别为 0 与截断上边界近似，单独记录上边界的解析截断偏差。
- **测试点**：固定 $256\times128$ 网格，以及独立的 Sobol 测试样本；对所有 A/B/C 模型完全一致，严禁混入训练配点。
- **D0 的数据对象**：`collocation.parquet`、`initial.parquet`、`boundary.parquet`、`test_grid.parquet`；至少包含 `s, tau, split, sample_id, seed`，边界/初始集另有 `target_u, condition_type`。

#### 1.3.2 D1：解析数据集（数值 ground truth）

对 D0 的**独立测试输入**生成 Black–Scholes 解析值：

$$
C(S,\tau)=S\Phi(d_1)-Ke^{-r\tau}\Phi(d_2),\quad
 d_{1,2}=\frac{\ln(S/K)+(r\pm\frac12\sigma^2)\tau}{\sigma\sqrt\tau}.
$$

$$
\Delta=\Phi(d_1),\qquad \Gamma=\frac{\varphi(d_1)}{S\sigma\sqrt\tau},\quad (S>0,\tau>0).
$$

- $\Phi$ 为标准正态 CDF，$\varphi$ 为其 PDF。$S=0$ 和 $\tau=0$ 使用解析极限/到期条件单独处理，不直接代入含 $\ln S$ 或 $1/\sqrt\tau$ 的公式。
- 除价格 RMSE/relative $L^2$ 外，独立报告 Delta 和 Gamma；Gamma 的常规精度评估优先使用 $\tau\ge0.05$ 等避开到期 kink 的区域。
- 为便于统一验证，价格使用无量纲 $u=C/K$；若导出实际价格则同时保存 `price_unit` 和 $K$，避免 $u$ 与货币报价混用。
- **禁止将解析价格误当作 PINN 内部 collocation 标签**；如进行监督学习对照，使用独立实验编号和一致的监督数据预算。

#### 1.3.3 D2：金融参数合成集（参数泛化测试）

建议第二阶段生成小规模、完全可复现的参数网格：

| 参数 | 示例取值 | 用途 |
|---|---|---|
| $\sigma$ | 0.10、0.20、0.35 | 波动率敏感度与参数外推 |
| $r$ | 0.00、0.03、0.06 | 利率敏感性 |
| $K$ | 40、50、60 | 执行价尺度变化 |
| $T$（年） | 0.25、0.50、1.00 | 不同到期期限 |
| $s=S/K$ | 0–3 | 统一无量纲状态域 |

**区分两类试验：**

1. **Independent-solve**：每组 $\mu=(r,\sigma,K,T)$ 分别求解一个 $w^*(\mu)$，量子储备池结构、编码规则固定；此实验验证跨参数重复求解成本，**不等于**单个网络对未见参数泛化。
2. **Parameterized PINN/QRC**：显式把 $\mu$ 纳入模型/储备池输入 $x=(s,\tau,\mu)$，联合训练一个参数化解算子，再用**未见过的参数组合**测试插值和外推。需为各模型提供相同的参数采样与监督/物理损失预算。

初期建议先做 Independent-solve，确认量子储备池具有可复用的特征表示；只有必要时再扩展输入编码维度。这里的合成数据不使用历史市场波动率，避免引入市场模型识别问题。

### 1.4 真实金融市场数据来源与实验定位

真实数据只在**市场适用性验证（M5-market pilot）**引入，不构成 M0–M4 的必要依赖。优先选用**Cboe SPX 欧式指数期权**：SPX/相关欧式系列为现金结算、欧式行权，而 SPY ETF 期权为美式行权，不能直接作为同一个欧式 BS PDE 的无条件真值。

| 来源 | 数据内容及可用性 | 推荐角色 | 官方入口 |
|---|---|---|---|
| **Cboe DataShop — Option EOD Summary** | 历史期权 EOD/15:45 快照、NBBO Bid/Ask、交易量、持仓量；可选 IV/Greeks，通常需要购买；官网提供样例 | 首选 SPX 报价试验 | https://datashop.cboe.com/option-eod-summary |
| **OptionMetrics IvyDB US** | 1996 年以来美股及指数期权历史价格、IV/Greeks、标的和股息等信息；按机构/商业授权获取 | 长时间序列、规模化校准 | https://optionmetrics.com/united-states/ |
| **FRED DGS1 等利率序列** | 美联储 H.15 一年期国债收益率等公开序列 | 初步利率参考与时间对齐 | https://fred.stlouisfed.org/series/DGS1 |
| **上交所数据服务** | 期权历史日内快照等，部分需商业订阅 | 国内金融市场扩展（非首期） | https://english.sse.com.cn/markets/dataservice/products/ |

**数据获取状态：**以上是经官方网页核查的候选来源，**并不表示本项目已经购买、下载或取得相应历史报价**。公开仓库可发布合成数据与生成脚本，真实商业行情仅在许可证允许的范围内保存和分享。

#### 1.4.1 市场试验的数据字段（建议统一 schema）

| 字段 | 含义及校验要求 |
|---|---|
| `quote_date`, `quote_timestamp`, `timezone` | 报价日期与准确快照时点，统一时区；不得混用收盘价与 15:45 报价 |
| `underlying_symbol`, `contract_symbol`, `option_type` | 标的、期权合约标识与 C/P 类型 |
| `exercise_style`, `settlement_type` | 行权类型、交收方式；首选欧式期权 |
| `strike`, `expiration_date`, `tau_years` | 执行价、到期日、按已声明 day-count 规则计算的剩余期限 |
| `underlying_spot`, `bid`, `ask`, `mid` | 同一时点的标的价格、报价区间和中间价；`mid=(bid+ask)/2` |
| `bid_size`, `ask_size`, `volume`, `open_interest` | 报价有效性和流动性指标（有则使用） |
| `risk_free_rate`, `dividend_yield`, `forward_price` | 与到期时长相匹配的贴现率、股息/远期假设及其来源 |
| `implied_vol` | 由一致的定价模型计算/官方提供；注明算法、是否剔除异常值 |
| `source`, `license`, `snapshot_id` | 来源、使用权限、抓取/采购批次；保证可追溯 |

数据接入建议先以**单日、单快照、有限到期月份的 SPX 欧式看涨期权**进行试点，避免把时间动态问题和跨执行价波动率曲面校准混在同一个实验中。

#### 1.4.2 清洗、对齐、切分与金融合理性验证

1. **时间对齐**：`bid/ask`、标的价格、利率曲线、股息/远期均对应同一估值日期；优先相同时点的标的价格和期权报价。若来源不能时间对齐，需披露估值偏差。
2. **报价有效性**：过滤缺失/非数值、`bid<0`、`ask<bid`、到期时间非正、明显过时及调整合约；`bid=0`、宽买卖价差和低成交量可单独标记与敏感性分析，不必一律删除。
3. **期限和价内程度**：首轮可限制剩余期限 7–180 天、`S/K` 在 0.8–1.2（**建议筛选窗口，非市场通用标准**），随后逐步放宽并报告过滤留存率。
4. **利率与分红**：FRED `DGS1` 只是单一期限的国债收益率代理，并非所有期限的精确零息贴现曲线；多期限市场拟合应改用期限匹配的折现因子/OIS/相应利率曲线。SPX 指数具有股息效应，需要股息收益率 $q$ 或期限匹配的远期信息。
5. **无套利筛查**：对同一快照、相同行权和到期结构核验非负性、执行价单调性、凸性及适用的 Put–Call Parity。存在 Bid/Ask 价差时应使用相容的不等式或容差，不能简单要求所有 mid 完全满足等式。
6. **防止数据泄漏**：单日横截面按执行价/到期合约块划分；多日数据按日期顺序训练/验证/测试，例如 60%/20%/20% 的时间块，而不是对相邻报价行随机切分。报告样本量、日期和合约范围。
7. **许可与审计**：商业原始报价不默认上传公开 GitHub；保存来源链接、数据字典、过滤规则、原始文件哈希、许可证和可重跑的处理脚本。

#### 1.4.3 真实市场数据不能直接验证固定系数 BS PDE

原始首期 PDE 假定常数 $r,\sigma$，并无股息项。若直接使用 SPX 真实行情，应至少在恒定股息收益率近似下改为：

$$
\partial_\tau V-\tfrac12\sigma^2S^2\partial_{SS}V-(r-q)S\partial_SV+rV=0,
$$

欧式看涨期权初始 payoff 不变。更重要的是，真实期权存在**波动率微笑与期限结构**，一组固定 $\sigma$ 无法一致拟合所有执行价和期限。

因此建议分成两个清晰的真实数据实验：

- **D3a：限定市场切片的定价合理性测试**：选取单日、窄价内程度和有限期限的欧式期权，以外部校准或局部近似的 $\sigma,r,q$ 比较三类数值模型与相应 BS 基准；结论仅限于**在给定模型假设下的求解能力**，不可将对真实 mid 的偏差全部归咎于 QRC。
- **D3b：真实波动率校准（独立扩展课题）**：在模型中显式加入待校准 $\sigma(K,T)$ 或换用局部/随机波动率模型，并增加市场报价损失及无套利约束。此时不再是首期的固定参数 BS-QRC-PINN，应单独制定模型定义、训练目标和验收条件。

例如，数据拟合可以定义为 $\mathcal L_{\rm data}=\sum_i\omega_i(\widehat C_i-\text{mid}_i)^2$；也可使用**报价区间惩罚**

$$
\mathcal L_{\rm quote}=\sum_i\omega_i\Bigl[\max(\text{bid}_i-\widehat C_i,0)^2+\max(\widehat C_i-\text{ask}_i,0)^2\Bigr].
$$

市场实验需同时报告价格 MAE/RMSE、IV RMSE（若可计算）、模型报价落入 Bid/Ask 的比例、无套利违反程度及时间块泛化表现。所有经典与 QRC 模型必须使用**相同的市场切分、相同特征维度预算和相同金融约束**。

### 1.5 数据目录、复现规范与数据阶段验收

建议在项目中将数据按许可证和用途分开，所有中间表保存为 Parquet/CSV，使用 `manifest.yaml` 记录源文件哈希、版本、转换脚本、日期、seed 和金融参数：

```text
data/
├── synthetic/                       # 可公开，随脚本生成
│   ├── d0_collocation/
│   ├── d0_boundaries/
│   ├── d1_analytic_reference/
│   └── d2_parametric/
├── market/                          # 商业行情：不默认入 Git
│   ├── raw/                         # 原始快照，受许可证保护
│   ├── processed/                   # 标准化行情、利率和远期
│   └── manifest.yaml
└── splits/                           # 统一 train/validation/test 清单
```

建议新增如下数据模块与测试：

```text
src/qrc_pinn/finance/
├── black_scholes.py
├── datasets.py
├── synthetic_generator.py
├── market_schema.py
├── market_cleaning.py
└── data_manifest.py
tests/
├── test_synthetic_reference.py
├── test_split_no_leakage.py
├── test_market_quote_validation.py
└── test_annualization_and_units.py
```

| 阶段 | 数据交付物 | 数据层验收标准 |
|---|---|---|
| M0 | D0 + D1（含解析价格/Greeks） | BS 解析极限、单位及边界条件测试通过；训练/测试独立 |
| M1 | 同一 D0/D1 的经典随机特征输入 | 所有经典基线共享测试集与误差定义 |
| M2 | 小规模 D0 + 缓存的量子特征与导数 | 每个 collocation ID 可追溯量子编码、模拟配置与导数步长 |
| M3 | D2 及噪声/参数敏感性实验 | 区分独立求解和参数化网络泛化，完整记录参数分割 |
| M4 | 20 原子特征缓存、时间和资源记录 | 不把昂贵输入点重复计入独立样本，资源可复现 |
| M5-market（可选） | D3 授权行情快照及清洗报告 | 行权类型匹配、无泄漏切分、时间/贴现/股息对齐、许可确认 |

> **数据层最终约束：**基础方法学论文的主结论应建立在 D0/D1 的可复现数学基准上；D2 检验参数变化下的能力，D3 检验真实市场适用性。不能用市场误差反证固定 QRC-PINN 求解 BS 方程的数值精度，也不能只依靠解析合成数据声称金融市场有效性。


## 2. 模型 A：经典 MLP-PINN

### 2.1 输入输出与训练目标

**输入：**$x=(s,\tau)$、PDE 系数、collocation 与边界/初始点。  
**输出：**价格近似 $u_\theta(x)$，训练产物为网络参数 $\theta^*$。  
**训练目标：**

$$
\mathcal J(\theta)=\lambda_r\frac{1}{N_r}\sum_{x_i\in D_r}|\mathcal Lu_\theta(x_i)|^2+
\lambda_0\frac{1}{N_0}\sum_{x_j\in D_0}|u_\theta(x_j)-g_0(x_j)|^2+
\lambda_b\frac{1}{N_b}\sum_{x_k\in D_b}|u_\theta(x_k)-g_b(x_k)|^2.
$$

实现：DeepXDE + PyTorch，前向 AD 计算 $\partial_\tau u,\partial_su,\partial_{ss}u$。典型网络 `2-[32/64]×4-1`，激活 `tanh`，Adam → L-BFGS。优先重跑官方 Black–Scholes 示例，再进行参数对照，避免一开始修改过多配置。

**M0 预期验收目标（建议值，非性能承诺）：**独立测试价格相对 $L^2$ 误差 $\le10^{-2}$；记录全域、近执行价、近到期的 Delta/Gamma 和残差分布。未达标先排查尺度、坐标方向、到期条件、采样和优化权重。

## 3. 模型 B：经典固定随机特征 PINN

### 3.1 结构

取固定随机 Fourier 或 RBF 特征 $\phi_C(x)\in\mathbb R^M$：

$$u_w(x)=\sum_{j=1}^Mw_j\phi_{C,j}(x).$$

仅训练线性读出权重 $w$；使用与量子特征相同的采样点、损失权重、特征维度，降低架构混杂因素。

### 3.2 线性最小二乘

构建 $A_{ij}=[\mathcal L\phi_j](x_i)$、$B_{kj}=\phi_j(x_k)$。收集初始和边界观测向量 $y_b$，求解：

$$\boxed{w^*=\arg\min_w\;\lambda_r\|Aw\|_2^2+\lambda_b\|Bw-y_b\|_2^2+\lambda_{\mathrm{reg}}\|w\|_2^2}. $$

在实现中按每类样本数归一化，建议优先 `scipy.linalg.lstsq` / QR / SVD 或 ridge 的稳定求解形式，不默认显式计算 $A^TA$（容易放大条件数）。

**重要判断：**若经典随机特征已达到目标误差，而 QRC 生成等量特征却成本更高，则量子实现缺乏端到端优势；不应只与 MLP 比较。

## 4. 模型 C：固定 Rydberg-QRC-PINN

### 4.1 量子特征输入、输出及固定参数

**金融输入：**$x=(s,\tau)$。  
**固定的量子配置：**$\kappa=(\{\mathbf r_i\},C_6,\Omega_0,\Delta_0,t_R,\text{编码系数})$。  
**量子输出：**测量估计的特征 $\phi_Q(x)$，非直接输出期权价格。  
**经典可训练输出：**$u_w(x)=w^T\phi_Q(x)$（常数特征并入 $\phi_Q$），最终得到 $w^*$。

采用具输入依赖控制的 Rydberg 动力学：

$$H_x(t)=\tfrac12\sum_i\Omega_i(t;x)X_i-\sum_i\Delta_i(t;x)n_i+\sum_{i<j}\frac{C_6}{|\mathbf r_i-\mathbf r_j|^6}n_in_j.$$

$$|\psi_x(t_R)\rangle=\mathcal T\exp[-i\int_0^{t_R}H_x(t)\,dt]|\psi_0\rangle.$$

“固定 reservoir”意为**编码规则、原子布局、物理模型及所有可优化控制参数不参与训练**；输入 $x$ 变化仍可使所施加脉冲发生变化。金融 $\tau$ 与物理时间 $t$ / 固定演化时长 $t_R$ 必须明确区分。

### 4.2 初始编码提议（待比较）

- **E1：局域失谐编码**：$\Delta_i(x)=\Delta_i^{(0)}+a_i f(s)+b_i g(\tau)$，$f,g$ 归一化并限定动态范围，$a_i,b_i$ 为固定系数。
- **E2：局域 Rabi 编码**：$\Omega_i(x)=\Omega_i^{(0)}+c_i f(s)+d_i g(\tau)$。
- **E3：前置编码脉冲 + 固定 reservoir 演化**：先执行 $U_{enc}(x)$，后使用完全不依赖 $x$ 的 $U_R$，方便单独辨析编码和多体演化的贡献。

优先 E1，再用 E3 进行消融。先做控制幅度无量纲化并遵守 Sagittarius 统一单位契约；不能将金融输入直接视作 MHz 或物理时间。

### 4.3 观测量与特征维度

以 $n_i=|r_i\rangle\langle r_i|$ 作为观测量，首期特征：

$$\phi_Q(x)=[1,\langle n_1\rangle_x,\ldots,\langle n_N\rangle_x,\{\langle n_i n_j\rangle_x\}_{(i,j)\in E}]^T.$$

- 第一阶段仅用 $N$ 个单体特征及常数项，可直接依赖公开的布居 API。
- 第二阶段加入近邻边集合 $E$ 的双体关联：如 20 原子链有 19 条近邻边，合计 **40 维**（1 + 20 + 19）。
- 第三阶段再比较全对关联（20 原子有 190 对，合计 211 维），但这可能增加测量与计算开销，需显式计价。
- 双体特征可由小规模状态概率分布或 bitstring 频率验证；必要时扩展 `ObservableAdapter`，避免直接假定现有公开接口已支持通用双体算符。

### 4.4 PDE 特征导数（首要研发瓶颈）

必须获得：

$$\phi_Q,\quad \partial_s\phi_Q,\quad\partial_{ss}\phi_Q,\quad\partial_\tau\phi_Q.$$

**D0：中心有限差分（最低可行路径）**

$$\partial_s\phi\approx\frac{\phi(s+h_s,\tau)-\phi(s-h_s,\tau)}{2h_s},$$
$$\partial_{ss}\phi\approx\frac{\phi(s+h_s,\tau)-2\phi(s,\tau)+\phi(s-h_s,\tau)}{h_s^2},$$
$$\partial_\tau\phi\approx\frac{\phi(s,\tau+h_\tau)-\phi(s,\tau-h_\tau)}{2h_\tau}.$$

在内部点，一次 PDE 残差可重用中心值，通常需要**5 次不同输入的 reservoir 演化**（中心、$s\pm h_s$、$\tau\pm h_\tau$），而不是 5 次重复 shots。靠近边界必须采用适当的一侧差分或避开残差采样边界；不能让移位输入越出金融域或脉冲允许范围。

**D1：更高精度路径**

- 小规模（1–4 原子）使用更紧积分容差/高阶差分建立导数真值。
- 根据 Julia 物理求解后端能力，评估 forward sensitivity ODE / 可微模拟器，而不是假定 Python autograd 可跨 Julia 和真实 QPU 自动工作。
- 对于依赖输入的哈密顿量，求解 $\partial_x|\psi\rangle$ 的伴随/灵敏度演化，必要时求二阶灵敏度；需额外求解成本。
- 同时记录有限差分误差、ODE 积分误差与 shots 统计误差；实验中单独报告每个贡献。

**统计风险：**有限 shots 下，二阶差分会把独立测量噪声按约 $1/h_s^2$ 放大；过小的 $h_s$ 不一定更准确。先在模拟器上用精确期望值验证，再注入 1,000 / 10,000 shots 的模拟测量噪声进行鲁棒性实验。

### 4.5 读出训练

用固定的量子特征替代 $\phi_C$，按第 3.2 节构造 $A,B$；由经典线性最小二乘得到 $w^*$。缓存的必须是**特征及其导数**，不建议为每个输入缓存完整量子态。

无套利与风险约束可作为下一阶段独立验证，必要时在 $w$ 空间加入线性不等式，例如 $\partial_s u\ge0$ 和 $\partial_{ss}u\ge0$ 在指定检查网格上的近似约束（可转成凸二次规划），但不要在 M2 首次闭环就引入过多目标。

## 5. 分阶段开发与验收

> 下表是建议研发顺序；周期为项目管理上的粗略计划，不是运行耗时预测，也不保证所有阶段必须达到 20 原子。

| 阶段 | 建议安排 | 重点任务 | 可交付物 | 进入下一阶段的门槛 |
|---|---|---|---|---|
| **M-1** 环境与求解器核验 | 第 1 周 | 固定版本、GPU doctor；CPU/GPU 物理一致性；验证 4/8/12/16/20 原子内存/时间 | `environment.json`, `solver_benchmark.csv` | 8 原子可靠复现；CUDA 失败时 CPU 回退可用 |
| **M0** 解析 + 经典 PINN | 第 2 周 | Black–Scholes 解析值、Greeks；DeepXDE/PyTorch MLP-PINN | `bs_reference.py`, `mlp_pinn.py`, 对比报告 | 价格 relative L2 ≤ 1e-2（首期建议目标） |
| **M1** 固定经典随机特征 | 第 3 周 | RFF/RBF、导数、稳定最小二乘、特征规模比较 | `random_feature_pinn.py`, `feature_fit.py` | 数值导数、最小二乘结果通过单元测试 |
| **M2** 4–8 原子量子闭环 | 第 4–5 周 | Sagittarius adapter、E1/E3 编码、单体特征、D0 导数、PDE 拟合 | `sagittarius_adapter.py`, `rydberg_features.py` | 全部输入→特征→导数→价格预测链路正确，可复现 |
| **M3** 8–12 原子性能验证 | 第 6–7 周 | 近邻双体关联、导数步长扫描、shots 噪声、消融实验 | `derivative_study.csv`, `ablation.csv` | 特征导数随求解精度收敛；测试价格、Greeks 稳定 |
| **M4** 16–20 原子扩展 | 第 8–9 周 | 状态存储、求解耗时、特征缓存、full/reduced、CUDA/CPU 比较 | `scaling.csv`, 性能剖析报告 | 20 原子运行可完成且结果可信；否则记录可复现上限 |
| **M5** 统一验证与论文预研 | 第 10 周及以后 | 统一测试、资源归因、误差/收益统计、技术报告 | `benchmark_report.md`, 图表、复现实验脚本 | 能给出可证伪、有置信区间的结论 |

### 5.1 M-1 环境核验命令

```bash
# 在已配置的 WSL2 / Ubuntu CUDA 环境中
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Sagittarius 源码仓库：使用公开文档提供的方式
cd Sagittarius/sagittarius_py
uv sync
uv run python -m juliapkg resolve
uv run sagittarius doctor
uv run python -m pytest tests/
```

可选检查（具体环境中 CUDA 后端可能初始化失败）：

```python
from sagittarius import doctor
print(doctor(backend="CUDA", initialize_backend=True))
```

项目应**把 Sagittarius 作为外部版本锁定依赖**，记录实际 Git commit；不要把研究实验代码混入模拟器核心仓库。GPU 后端是否真的可用以本机诊断和实际测试为准，不由 `use_gpu=True` 配置自行证明。

### 5.2 M2 可执行的调用顺序（伪代码）

```python
# 本段为接口设计伪代码，不是现有 Sagittarius API 的直接可执行示例。
register, fixed_params = build_rydberg_register(n_atoms=4)
encoder = FixedInputEncoder(fixed_params)
reservoir = SagittariusAdapter(register, fixed_params)

# x 是归一化金融坐标 (s, tau)；tau 并非物理仿真时间

def quantum_features(x):
    pulses = encoder.encode(x)     # 输入改变脉冲，但编码系数保持固定
    return reservoir.measure_features(pulses, observables="populations")

for x in collocation_points:
    phi[x] = quantum_features(x)
    dphi_ds[x], d2phi_ds2[x], dphi_dtau[x] = finite_differences(
        quantum_features, x, steps=(h_s, h_tau)
    )

A = assemble_black_scholes_residual_matrix(phi, dphi_ds, d2phi_ds2, dphi_dtau)
B, y = assemble_boundary_matrix_and_targets(...)
w = regularized_least_squares(A, B, y, lambda_reg=...)
report = evaluate_analytic_price_greeks_and_costs(w, ...)
```

## 6. 实验矩阵与计算预算

### 6.1 初始实验矩阵（避免全因子爆炸）

| 实验组 | 原子数 | 特征集合 | collocation 点 | 目的 |
|---|---:|---|---:|---|
| Q0 | 4 | 单体 | 16、32、64 | API、编码、导数正确性 |
| Q1 | 8 | 单体；单体+邻近双体 | 32、64、128 | 验证特征与拟合收益 |
| Q2 | 12 | 单体+邻近双体 | 64、128、256 | 小规模收敛与噪声分析 |
| Q3 | 16 | 单体+邻近双体 | 32、64、128 | 性能、求解器与显存扩展 |
| Q4 | 20 | 单体+邻近双体 | 16、32、64 起步 | 20 原子里程碑；依据实际单次求解耗时再扩展 |

经典 A/B 基线也要在 quantum 的小样本条件下做同数据比较，另设充分数据训练版本；应**同时报告等 collocation 点数与等总计算预算**两套结果。

### 6.2 资源预估和约束

单个 `complex128` 纯态的理论状态内存：

| 原子 N | 维度 $2^N$ | 单份状态向量 |
|---:|---:|---:|
| 8 | 256 | 4 KiB |
| 12 | 4096 | 64 KiB |
| 16 | 65536 | 1 MiB |
| 20 | 1048576 | 16 MiB |

但模拟时间成本、积分器临时分配、哈密顿量的实施形式、传输与初始化开销才是瓶颈。优先使用矩阵自由 / 稀疏算符路径；使用 blockade 约束空间时需核实被删去的物理态是否影响 QRC 特征表达。不得为了规模而把 reduced-basis 结果当作 full-space 的无误差等价物。

**成本计数：**完整评估至少分成：初始化 Julia/CUDA、输入脉冲生成、状态演化、观测量计算、导数估计、组装最小二乘、拟合、测试。记录 wall time、峰值 GPU 显存、CPU RAM、求解器误差容限、设备利用率。对于真实量子硬件扩展，还需记录 shots、制备/测量与排队成本。

## 7. 评估指标与验收口径

**金融精度：**

1. 价格相对 $L^2$：$\|V_{pred}-V_{BS}\|_2/\|V_{BS}\|_2$；辅以价格 MAE、最大绝对误差和 $\mathrm{RMSE}/K$。
2. Delta：$\partial_SV$ 误差；Gamma：$\partial_{SS}V$ 误差。Gamma 单独报告 $\tau\ge0.05$ 等避开 kink 的区域。
3. PDE 残差：独立测试网格的均值、95% 分位数及最大值；**残差小不等于价格精度高**。
4. 金融一致性：$V\ge0$、$\Delta\in[0,1]$、$\Gamma\ge0$ 的离散违反比例和违反幅度（在合理容差下）。

**数值和硬件：**

5. 特征导数随 $h_s,h_\tau$ 和积分容差变化的收敛曲线。
6. 不同原子数 $N$、可用特征数 $M$ 的价格/Greeks 误差—成本曲线。
7. `CPU vs CUDA`、`full vs blockade reduced` 的速度和物理偏差。
8. 精确期望值 vs 有限 shots 的鲁棒性；量化二阶差分噪声放大。
9. 首次部署/训练成本、缓存特征后再训练成本、单次定价推理成本，分别报告。
10. 各模型固定 5 个种子（昂贵实验如无法重复必须显式标明），报告均值和标准差。

**建议判定门槛（提议，不是理论保证）：**

- **M0 合格**：价格相对 $L^2\le10^{-2}$，价格和 Greeks 与解析公式一致；所有训练配置可复现。
- **M2 合格**：4–8 原子 QRC-PINN 在完整 PDE/边界损失下稳定输出价格；导数对精度参数有可解释的收敛趋势。
- **M4 合格**：20 原子在本机完成至少一轮完整的“输入→动力学→特征→导数→读出→验证”运行，并记录真实成本；不以胜过经典基线作为 20 原子功能验收前提。
- **研究结论层面**：仅当 QRC 在同等误差/预算下优于最强经典随机特征基线，才讨论特征表达收益；若要进一步声称量子加速，必须说明经典模拟开销、可用硬件、测量成本，以及与真正经典算法的成本归一化方式。

## 8. 风险、降级路径与停止条件

| 风险 | 具体表现 | 降级或处理 |
|---|---|---|
| CUDA 后端不稳定 | 编译/驱动不匹配、GPU kernel 报错 | 4–8 原子先用 CPU；锁定 Sagittarius 和 Julia/CUDA 版本，单独排查 CUDA |
| 20 原子调用过慢 | 虽能保存状态但每点求解耗时高 | 从 16 点开始；缓存特征；避免全参数扫描；限制到 12–16 原子也能形成论文方法结论 |
| 量子特征退化 | 多个观测量强相关、数值秩低 | 特征标准化、SVD、ridge；改变输入编码、脉冲时长、几何图；与 RFF 基线比较 |
| 二阶导数不稳定 | $h_s$ 过小放大噪声，过大有截断误差 | 步长扫描、灵敏度方程、平滑近似和数据质量审计 |
| blockade 压缩引入偏差 | 物理动力学差异影响 feature | 小规模 full/reduced 对照；报告 blockade 截断误差 |
| loss 权重病态 | PDE 残差小，但价格错误 | 输入/输出无量纲化；QR/SVD；验证全域；调整边界权重/硬约束 |
| 误导性加速结论 | 只比较 MLP 训练时长，不算量子特征生成 | 全成本报告，必须与 Random Feature PINN 比较 |

**Go/No-Go 决策：**

- M1 若经典随机特征在很小维度下已经充分优于 MLP，仍继续 QRC，但研究重点变为**同特征维数、同预算**的特征表达比较。
- M2 若 8 原子无法可靠计算二阶导数，暂缓扩大原子数，先研究导数估计或转用弱形式/变分形式的 PINN。
- M3 若 QRC 特征列高度相关且无法通过编码与布局改进，停止盲目增大 N，优先改进特征及采样设计。
- M4 若 20 原子性能代价过高，允许以 12/16 原子完成方法论文，并将 20 原子性能评估作为独立工程报告。

## 9. 仓库、模块与里程碑管理

以当前项目 `rydberg-qrc-pinn` 为研究仓库根目录，Sagittarius 保持独立 SDK，仅以锁定版本及 Git commit 的外部依赖接入。下列为目标结构：标注“已有”的文件对应当前项目，其余按第 5 节里程碑逐步建设；Python 子包的 `__init__.py` 在树中省略。

```text
rydberg-qrc-pinn/
├── README.md
├── AGENTS.md                         # 已有：项目 AI 规范与决策卡入口
├── pyproject.toml                    # Python 包、依赖与测试配置
├── uv.lock                           # 锁定 Python 依赖及 Sagittarius 来源版本
├── .gitignore                        # 本地数据、缓存与大体积运行产物规则
├── skills/
│   └── scientific-decision-card/
│       └── SKILL.md                  # 已有：任务执行前七项决策卡
├── docs/
│   ├── plan/
│   │   └── Rydberg_QRC_PINN_Research_Plan.md  # 已有：研究实施方案
│   ├── requirements/                 # 科学目标、实验范围与验收条件
│   ├── reports/
│   │   ├── benchmark_report.md          # M5：方法结论、适用范围与复现入口
│   │   ├── figures/                     # 误差、导数收敛、消融与扩展曲线
│   │   └── tables/                      # solver_benchmark/derivative_study/ablation/scaling.csv
│   └── specs/                        # 数据、接口、物理单位与缓存契约
├── configs/
│   ├── black_scholes.yaml            # 金融参数、无量纲域与初边值条件
│   ├── datasets.yaml                 # D0/D1 采样、划分、seed；可选 D2 参数网格
│   ├── classical.yaml                # MLP、RFF/RBF 与读出正则化配置
│   ├── reservoir_4.yaml              # M2：4 原子最小闭环
│   ├── reservoir_8.yaml              # M2/M3：8 原子对照
│   ├── reservoir_12.yaml             # M3：12 原子精度与噪声实验
│   ├── reservoir_16.yaml             # M4：16 原子扩展
│   ├── reservoir_20.yaml             # M4：20 原子目标配置，需实测验收
│   ├── solver.yaml                   # 求解器、容差、CPU/CUDA、full/reduced
│   ├── derivatives.yaml              # 差分步长、边界策略与收敛扫描
│   ├── benchmark.yaml                # 公共测试集、种子、资源预算与消融矩阵
│   └── market.yaml                   # 可选 M5-market：市场切片、清洗与校准设定
├── data/
│   ├── synthetic/                    # 第 1.5 节：可复现生成的合成数据
│   │   ├── d0_collocation/           # 内部配点，无价格监督标签
│   │   ├── d0_boundaries/            # 初始条件与边界条件点及约束目标
│   │   ├── d1_analytic_reference/    # 独立评估价格、Delta、Gamma
│   │   ├── d2_parametric/            # 可选：参数化合成数据
│   │   └── manifest.yaml             # 生成版本、参数、seed、文件哈希与来源
│   ├── splits/                       # 统一划分清单、独立测试坐标与参数分割
│   └── market/                       # 可选 D3：依授权使用，不默认入 Git
│       ├── raw/                      # 原始授权行情快照
│       ├── processed/                # 清洗后报价与对齐的利率、股息/远期
│       └── manifest.yaml             # 数据来源、许可、快照与处理记录
├── src/qrc_pinn/
│   ├── finance/
│   │   ├── black_scholes.py          # 解析价格、Greeks、PDE、初边值与尺度转换
│   │   ├── datasets.py               # 配点、初边值点、数据划分与加载
│   │   ├── synthetic_generator.py    # D0/D1 生成；可选 D2 参数扫描
│   │   ├── data_manifest.py          # 数据版本、哈希、来源与划分追溯
│   │   ├── market_schema.py          # 可选：第 1.4 节市场字段与单位契约
│   │   └── market_cleaning.py        # 可选：报价校验、时间对齐与分组切分
│   ├── classical/
│   │   ├── mlp_pinn.py               # 模型 A：DeepXDE/PyTorch MLP-PINN
│   │   ├── random_features.py        # 模型 B：固定 RFF/RBF 特征及导数
│   │   └── fdm.py                    # 按需实现：FDM 数值交叉检查基线
│   ├── quantum/
│   │   ├── sagittarius_adapter.py    # Python/Julia SDK、设备与求解器适配
│   │   ├── encoding.py               # 固定金融输入→脉冲规则，含单位与范围检查
│   │   ├── observables.py            # 单体布居；经小规模验证后加入双体关联
│   │   ├── rydberg_features.py       # 模型 C：组合编码、演化与观测，提供特征映射
│   │   ├── derivatives.py            # 首期有限差分；后续评估灵敏度方案
│   │   ├── noise.py                  # M3：有限 shots 模拟与导数噪声分析
│   │   └── cache.py                  # 特征及导数缓存，不默认缓存完整量子态
│   ├── training/
│   │   ├── residual_matrix.py        # 经典/量子特征共用的 PDE 与初边值矩阵
│   │   └── least_squares.py          # QR/SVD/ridge；仅优化固定特征的经典读出
│   └── benchmark/
│       ├── metrics.py               # 价格、Greeks、残差、金融一致性与统计汇总
│       ├── profiler.py              # 分阶段 wall time、RAM、显存与缓存命中
│       └── report.py                # 等配点/等预算对照、误差—成本曲线与报告
├── scripts/
│   ├── generate_synthetic_data.py    # 按配置生成 D0/D1；可选 D2 与划分清单
│   └── prepare_market_data.py        # 可选：处理已授权 D3 数据，输出清洗报告
├── tests/
│   ├── test_finance_reference.py     # BS 解析极限、价格与 Greeks
│   ├── test_synthetic_reference.py   # 第 1.5 节：合成参考数据正确性
│   ├── test_split_no_leakage.py      # 训练/验证/测试与参数分组隔离
│   ├── test_annualization_and_units.py  # 年化、无量纲化与单位转换
│   ├── test_random_feature_derivatives.py  # RFF/RBF 导数对照
│   ├── test_encoding.py              # 金融坐标与物理脉冲的单位、范围
│   ├── test_observables.py           # 小规模单体/双体关联对照
│   ├── test_derivative_1atom.py      # 1 原子可解析案例与差分收敛
│   ├── test_residual_matrix.py       # PDE、初边值矩阵与样本数归一化
│   ├── test_least_squares.py         # 稳定求解、正则化与读出复现
│   ├── test_cache.py                # 配置/输入/步长变化时的缓存有效性
│   ├── test_sagittarius_cpu_gpu.py   # 环境具备时核验 CPU/CUDA 一致性
│   └── test_market_quote_validation.py  # 可选：市场报价与金融合理性
├── experiments/
│   ├── m_minus1_runtime.py           # M-1：环境、CPU/GPU 与求解器基准
│   ├── m0_mlp_baseline.py            # M0：解析基线与 MLP-PINN
│   ├── m1_random_features.py         # M1：固定经典随机特征与最小二乘
│   ├── m2_rydberg_4_8.py             # M2：4–8 原子闭环
│   ├── m3_derivative_noise.py        # M3：步长、容差、shots、观测与编码消融
│   ├── m3_parameter_generalization.py  # 可选 D2：独立求解/参数化模型分别评估
│   ├── m4_scaling_to_20.py           # M4：12/16/20 原子、full/reduced、CPU/CUDA
│   ├── m5_benchmark.py               # M5：统一误差、资源与重复试验统计
│   └── m5_market_pilot.py            # 可选 D3：D3a 市场切片；D3b 另立实验设定
└── artifacts/                       # 本地运行产物；大文件不默认提交
    ├── cache/                       # 按输入、物理配置、求解器与差分配置索引
    └── runs/
        └── <run_id>/                # 一次固定配置、数据划分与种子的独立试验
            ├── config.yaml          # 本次展开后的完整配置
            ├── environment.json     # 硬件、Python/Julia/CUDA 与 Sagittarius commit
            ├── manifest.yaml        # 数据/划分哈希、缓存引用与产物清单
            ├── metrics.json         # 误差、资源成本与重复次数记录
            └── model/               # MLP 权重或固定特征的经典读出权重
```

**模块与产物边界：**

- `finance/` 统一数学问题和数据契约，`classical/`、`quantum/` 分别提供模型或特征；`training/` 供两类固定特征共用，MLP 的优化由 `mlp_pinn.py` 负责。Sagittarius 的适配与本项目特征组合在 `quantum/` 内完成，具体上游接口以锁定版本核验结果为准。
- `scripts/` 负责数据准备，`experiments/` 负责按里程碑组织实验，复用 `src/qrc_pinn/` 的实现。第 5 节交付物中的 `bs_reference.py` 对应本结构的 `finance/black_scholes.py`；`random_feature_pinn.py`、`feature_fit.py` 的职责由 `classical/random_features.py` 与 `training/` 承担，避免重复维护同一实现。
- `data/` 保存数据与固定划分，`artifacts/cache/` 保存可重用的特征和导数，`artifacts/runs/<run_id>/` 保存一次试验的复现信息及原始结果；`docs/reports/` 汇总可分享的统计表和图。运行记录需区分首次特征生成、缓存后拟合与推理成本，数据和缓存均通过 manifest 追溯。
- D0/D1 是 M0–M5 主线；D2 参数泛化和 D3 市场模块按需引入。原子数、CPU/CUDA、full/reduced 是待验证的配置维度，按第 6 节选择实验组合。市场数据遵循第 1.4 节许可与用途约束，公开产物须排除未经许可的商业原始数据。

**GitHub milestone 建议：**

- `M-1 Runtime and solver validation`
- `M0 Black-Scholes classical baseline`
- `M1 Classical random-feature baseline`
- `M2 Rydberg feature and PDE derivatives`
- `M3 QRC-PINN accuracy and ablations`
- `M4 Scale to 20 atoms`
- `M5 Benchmark and paper-ready report`

每个 Issue 记录 `Objective / Input / Output / Acceptance / Evidence`；不引入第二套项目管理平台。

## 10. 下一步最小可执行任务

优先完成以下任务，避免直接进入 20 原子大规模训练：

- [ ] 记录 `Sagittarius` commit、Python/Julia/CUDA、GPU 驱动版本，跑通 CPU 测试和 CUDA doctor。
- [ ] 建立 BS 解析价格、Delta、Gamma 参考函数和单位测试。
- [ ] 复现 DeepXDE 官方 Black–Scholes PINN；输出解析解对照图、误差表与训练时间。
- [ ] 用 `scipy.linalg.lstsq` 构建 RBF / RFF 随机特征 PINN；输出同格式报告。
- [ ] 基于 Sagittarius 4 原子搭建 `feature(s,tau)`；首先只提取 $\langle n_i\rangle$。
- [ ] 编写并验证 `\partial_s\phi,\partial_{ss}\phi,\partial_\tau\phi`；核查 1 原子可解析案例。
- [ ] 4 原子模型形成首个完整最小二乘 PDE 训练闭环。
- [ ] 扩到 8 原子并与 RFF 模型对照后，再决定如何扩到 12/16/20。

## 参考文献与官方资料

1. **Sagittarius**. *Research SDK for classical simulation of Rydberg neutral-atom analog dynamics*. 仓库与技术文档：https://github.com/Qosmos-Technology/Sagittarius 。核查日期：2026-11-17。
2. **Sagittarius Solver Configuration Contract**. `SPEC-API-005`, https://github.com/Qosmos-Technology/Sagittarius/blob/main/docs/api/SPEC-API-005-solver-configuration.md 。
3. **DeepXDE**. *Black-Scholes Equation* (官方教程), https://deepxde.readthedocs.io/en/stable/demos/pinn_forward/black_scholes.html 。
4. **Raissi, M., Perdikaris, P., Karniadakis, G. E.** (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics. https://doi.org/10.1016/j.jcp.2018.10.045 。
5. **Gonon, L.** (2023). *Random Feature Neural Networks Learn Black-Scholes Type PDEs Without Curse of Dimensionality*. Journal of Machine Learning Research 24(189):1–51. https://www.jmlr.org/papers/v24/21-0987.html 。注意其学习设定与本文的 PDE residual 特征最小二乘并非完全相同。
6. **Dees, J., Jacquier, A., Laizet, S.** (2024). *Unsupervised random quantum networks for PDEs*. Quantum Information Processing 23, 325. https://doi.org/10.1007/s11128-024-04537-0 。
7. **Kyriienko, O., Paine, A. E., Elfving, V. E.** (2021). *Solving nonlinear differential equations with differentiable quantum circuits*. Physical Review A 103, 052416. https://doi.org/10.1103/PhysRevA.103.052416 。
8. **Liu et al.** (2026). *Practical quantum reservoir computing in Rydberg atom arrays*. Physical Review A 113, 042401. https://doi.org/10.1103/pkhd-pl3w 。有关单步 QRC 噪声稳定性的数值研究，不是金融 PINN 验证。


## 官方数据来源

9. **Cboe DataShop**. *Option EOD Summary*（期权 EOD / 15:45 NBBO 报价与样本说明）. https://datashop.cboe.com/option-eod-summary 。核查日期：2026-11-17。
10. **Cboe Global Markets**. *S&P 500 Index Options*（SPX 欧式行权与现金结算）. https://www.cboe.com/tradable-products/sp-500/spx-options/ 。核查日期：2026-11-17。
11. **Cboe Global Markets**. *S&P 500 Weeklys Options: Product Specifications*（SPX/SPXW/SPY 行权风格区分）. https://www.cboe.com/tradable_products/sp_500/spx_weekly_options/specifications 。核查日期：2026-11-17。
12. **OptionMetrics**. *IvyDB US: Historical Options and IV Data*. https://optionmetrics.com/united-states/ 。核查日期：2026-11-17。
13. **Federal Reserve Bank of St. Louis / Board of Governors**. *FRED Series DGS1: Market Yield on U.S. Treasury Securities at 1-Year Constant Maturity*. https://fred.stlouisfed.org/series/DGS1 。核查日期：2026-11-17。
14. **Shanghai Stock Exchange**. *Historical Market Data Products*（含股票期权日内历史快照）. https://english.sse.com.cn/markets/dataservice/products/ 。核查日期：2026-11-17。

---

**项目最终验收标准：**形成可复现的完整研究闭环与独立结果：`Black–Scholes analytic → MLP-PINN → classical fixed features → Sagittarius 4/8/12/16/20 atom features → derivative validation → linear least squares → price/Greeks/cost benchmarks`。论文结论必须明确区分数学可行性、模拟器性能、未来 QPU 潜在优势以及任何尚未实证的量子加速假设。
