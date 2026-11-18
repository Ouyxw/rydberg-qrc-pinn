# Rydberg-QRC-PINN

基于 Sagittarius 的固定 Rydberg 量子储备池特征与 PINN 研究项目，首期以欧式看涨期权 Black–Scholes PDE 为数学基准，比较 MLP-PINN、经典固定随机特征和固定量子特征的误差与计算成本。

当前状态：仓库目录和文件骨架已初始化。Python 模块、实验入口与测试文件仅包含职责说明；YAML 配置为待配置模板。算法、测试用例和实验结果将在相应里程碑中实现、验证与记录。

## 项目入口

- [研究实施方案](docs/plan/Rydberg_QRC_PINN_Research_Plan.md)：数学设定、实验矩阵、验收条件和第 9 节仓库结构。
- [AI 工作规范](AGENTS.md)：先回答七项决策卡并提交方案，经人工明确确认后执行。
- [科学决策卡 skill](skills/scientific-decision-card/SKILL.md)：必答问题、范围说明和方案模板。

## 目录职责

| 路径 | 职责 |
| --- | --- |
| `src/qrc_pinn/finance/` | Black–Scholes 数学定义、数据生成与数据契约 |
| `src/qrc_pinn/classical/` | MLP-PINN、RFF/RBF 与按需引入的 FDM 基线 |
| `src/qrc_pinn/quantum/` | Sagittarius 适配、固定输入编码、观测、导数与缓存 |
| `src/qrc_pinn/training/` | 固定经典/量子特征共用的残差矩阵与最小二乘读出 |
| `src/qrc_pinn/benchmark/` | 精度、成本、重复试验统计与报告 |
| `configs/` | 按主题和原子规模划分的待配置 YAML 骨架 |
| `data/` | D0/D1 主线数据、固定划分及可选 D2/D3 数据 |
| `scripts/` | 数据准备入口骨架 |
| `experiments/` | M-1 至 M5 及可选参数泛化、市场实验入口骨架 |
| `tests/` | 待实现的数学、数据与数值链路验证 |
| `docs/requirements/`、`docs/specs/` | 后续需求、验收条件与接口契约 |
| `docs/reports/` | 基准报告、统计表和图 |
| `artifacts/cache/`、`artifacts/runs/` | 本地缓存及实际运行时生成的试验记录 |

## 数据与运行记录

内部 PDE 配点不带价格监督标签；解析价格和 Greeks 默认用于独立评估。D2 参数化合成数据和 D3 市场数据为可选扩展。数据 manifest 当前为待生成模板，真实数据产生后再填写来源、参数、随机种子和文件哈希。

数据负载与运行产物按 `.gitignore` 管理；空目录通过 `.gitkeep` 保留，数据 manifest 模板可进入版本管理。市场数据的使用和分享遵循研究计划中的许可要求。

`artifacts/runs/<run_id>/` 是未来实际试验的目录约定。每次运行再写入配置、环境、数据与产物 manifest、指标和模型权重；目录初始化阶段不生成试验实例或测量结果。

## 后续开发

先执行研究计划 M-1，核验 Python/Julia/CUDA 与 Sagittarius 的实际版本和运行能力，再补充依赖并真实生成 `uv.lock`。`pyproject.toml` 当前只包含最小包元数据、src 布局和测试路径设置。

各里程碑按研究计划逐步实现。当前配置和入口尚不足以启动训练；本次初始化验收为结构、语法和格式检查，空测试骨架不代表功能测试通过。
