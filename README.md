# dachuang_cnn_lstm

大创项目「基于 CNN-LSTM 的巧家地震震前水氡-水位双流体联合建模」。

> 状态：阶段 0（环境搭建）✅ → 阶段 1（数据管道）✅ → 阶段 2（基线复现）✅ → 阶段 3（CNN-LSTM）✅ → 阶段 4（异常检测+干扰排除）✅ → 多变量实验 ✅ → Web 可视化 ✅

## 目录

```
dachuang_cnn_lstm/
├── README.md           ← 本文件
├── AGENTS.md           ← agent/新协作者工作指引（数据格式陷阱、既定技术决策）
├── requirements.txt    ← 依赖及版本（Python 3.14.5，torch 2.13.0+cpu）
├── smoke_test.py       ← 合成数据端到端冒烟测试（python smoke_test.py）
├── run_phase2.py       ← 阶段2基线复现（西昌32井水氡，11年长序列）
├── run_phase3.py       ← 阶段3 CNN-LSTM训练（西昌32井水氡）
├── run_phase4.py       ← 阶段4 残差LOF异常检测+干扰排除+震例对应
├── run_multi_var.py    ← 多变量批量训练（水氡/水位/双变量/多变量×6窗口）
│
├── web/                ← Web可视化平台（FastAPI + ECharts）
│   ├── app.py          ← 启动入口（python app.py）
│   ├── api/            ← 后端API（数据/模型/异常）
│   └── static/         ← 前端页面（HTML + JS + CSS）
│
├── src/                ← 源码
│   ├── common.py       ← set_seed / load_paths / load_config
│   ├── data/           ← 阶段1数据管道：parse / clean / merge / split
│   ├── features/       ← 滑窗构造、zscore、去年变
│   ├── models/         ← baselines(naive/RF)、cnn_lstm(主模型)、lstm_rf(文献复现)、evaluate
│   ├── anomaly/        ← lof_detect、events(事件聚合+互证)、sensitivity
│   ├── viz/            ← figures（对齐文献图版式，300dpi）
│   └── analysis/       ← interference(干扰归因)、molchan(效能检验)——存根，等数据
│
├── configs/            ← 实验配置 JSON
│   └── paths.json      ← 数据路径（external=原始数据只读，internal=产物）
│
├── data/clean/         ← 清洗后数据（程序生成，禁止手改）
│   ├── long.csv        ← 统一长表（66,034行，110条序列）
│   └── dataset_info.csv← 序列级质量一览表
│
├── results/            ← 每次实验一个子目录
│   ├── pipeline/       ← 阶段1产物
│   ├── smoke/          ← 冒烟测试产物
│   ├── phase2_baselines/ ← 阶段2基线结果
│   └── phase3_cnn_lstm/ ← 阶段3 CNN-LSTM结果
│
├── figures/            ← 出版级图件（300 dpi）
│
├── docs/               ← 技术文档
│   ├── new/            ← 整合后的文档（数据方案.md、技术方案.md）
│   ├── Web可视化计划.md ← Web可视化设计方案
│   └── ...
│
├── notes/              ← 阶段执行笔记（每阶段完成一篇）
│   └── ...
│
├── papers/             ← 本版本文献 PDF 及精读笔记
│
└── proposal/           ← 大创申请书
```

## 快速开始

```bash
# 环境
D:\桌面\地质\大创\dataAnalysis\.venv\Scripts\python.exe

# 冒烟测试（合成数据端到端验证）
python smoke_test.py

# 阶段2基线复现（西昌32井水氡，11年长序列）
python run_phase2.py

# 阶段3 CNN-LSTM训练（西昌32井水氡）
python run_phase3.py

# 阶段4 残差LOF异常检测+干扰排除+震例对应
python run_phase4.py

# 数据管道（重新生成 data/clean/long.csv）
python src/data/run_pipeline.py

# 启动Web可视化平台
cd web && python app.py
# 浏览器访问 http://localhost:8002
```

## Web 可视化平台

基于 **FastAPI + ECharts** 的交互式可视化界面，用于展示数据、模型预测和异常检测结果。

### 功能

| 页面 | 功能 |
|------|------|
| **数据概览** | 台站选择、测项选择、时序曲线、数据质量统计 |
| **模型预测** | 变量选择（水氡/水位/双变量/多变量）、窗口选择（15-120天）、模型对比、残差图、评估指标表 |
| **异常检测** | 异常得分时序、异常事件列表、敏感性分析 |

### 数据源

| API | 读取文件 |
|-----|----------|
| `/api/stations` | `data/clean/long.csv` |
| `/api/data/{id}` | `data/clean/long.csv` |
| `/api/predictions/*` | `results/phase*/` 下的 metrics.json + predictions_test.csv |
| `/api/variables` | `results/radon/`, `results/level/` 等 | 可用变量组合列表 |
| `/api/predictions/{var}/{window}` | `results/{var}/window_{n}/` | 指定变量+窗口的预测结果 |
| `/api/multi_var_comparison` | `results/multi_var_comparison.json` | 多变量对比汇总 |
| `/api/anomalies/*` | `results/smoke/events.csv` |

### 启动

```bash
cd web
python app.py
# 访问 http://localhost:8002
```

## 数据路径

原始数据在 `D:\桌面\地质\大创\dataAnalysis\` 下：
- `OriData/`：云南 39 台站流体日值（127 个 TXT）
- `OriData2/`：整理后台站 xlsx（含西昌32井 11 年水氡）
- `OriData3/`：西昌气象三要素 + 地震目录（USGS 819 事件）

配置集中在 `configs/paths.json`，代码零硬编码。

## 当前进度

| 阶段 | 状态 | 产出 |
|------|------|------|
| 0 环境搭建 | ✅ | requirements + 目录骨架 + 种子复现验证 |
| 1 数据管道 | ✅ | data/clean/long.csv（66,034行，110条序列） |
| 2 基线复现 | ✅ | RF(0.6031) < LSTM(0.6150) < LSTM-RF(0.6589)，排序与文献一致 |
| 3 CNN-LSTM | ✅ | CNN-LSTM(0.6748) > LSTM-RF(0.6126)，卷积层带来增益 |
| 4 异常检测+干扰排除 | ✅ | 3个事件全部通过三道关（非气象+震前窗口有地震） |
| 5 多变量实验 | ✅ | 水氡单变量仍最优（R²=0.6895），水位/气象特征反而降低精度 |
| 6 Web可视化 | ✅ | http://localhost:8002（变量/窗口/模型选择，实时刷新） |

## 三条纪律

1. **原始数据不可变**：`D:\桌面\地质\大创\dataAnalysis\OriData*` 只读
2. **先跑通再调优**：新模块先在西昌32井单变量上端到端验证
3. **结果必须落盘**：没有写进 `results/` 的实验等于没做过
