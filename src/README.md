# src/ — 源码

所有模块通过 `src.common` 统一管理种子和路径，禁止硬编码。

---

## common.py — 通用工具

| 函数 | 用途 |
|------|------|
| `set_seed(seed)` | 固定 random/numpy/torch 随机种子，所有实验入口必须调用 |
| `load_paths()` | 读取 `configs/paths.json`，返回绝对路径字典（external 只读，internal 自动建目录） |
| `load_config(name)` | 读取 `configs/<name>.json` 实验配置 |

---

## data/ — 阶段1 数据管道

一次性写好、全程复用。产出 `data/clean/long.csv` + `dataset_info.csv`。

| 文件 | 功能 |
|------|------|
| `parse.py` | 三来源解析：OriData TXT(GBK) + 元数据表(伪xlsx) + OriData2 xlsx(混杂日期格式)；小时值宽表自动熔融 |
| `clean.py` | 缺测处理：999999→NaN；≤7天插值；hourly 聚合为日均值(标记agg)；>20%缺测标 low_quality |
| `merge.py` | 元数据 join + OriData2 去重（同测项大类内数值匹配，跨类禁止） |
| `split.py` | 序列级质量统计 + 时间顺序 8:2 划分 |
| `run_pipeline.py` | 总入口（幂等），`python src/data/run_pipeline.py` 一键运行 |

---

## features/ — 特征工程

| 文件 | 功能 |
|------|------|
| `build.py` | `zscore_fit/apply`（训练段拟合，防泄漏）、`make_windows`（单变量滑窗）、`make_windows_multi`（多变量滑窗）、`remove_annual_cycle`（去年变） |

---

## models/ — 预测模型

| 文件 | 模型 | 配置 |
|------|------|------|
| `baselines.py` | `naive_predict`（持久性）、`fit_rf/predict_rf`（随机森林） | RF: 100树、max_depth=100、min_leaf=5（文献同款） |
| `cnn_lstm.py` | `CNNLSTM`（主模型）+ `train_torch`（通用训练循环）+ `predict_torch` | Conv1D(64,k3)→Pool→Conv1D(32,k3)→LSTM(64)→Dropout→Dense |
| `lstm_rf.py` | `fit_lstm_rf/predict_lstm_rf`（LSTM-RF 级联） | LSTM 隐状态→RF 回归（文献最优方法） |
| `evaluate.py` | `rmse/mae/mape/r2` 指标 + `save_run` 实验落盘 | 与文献公式逐一对齐 |

---

## anomaly/ — 异常检测

| 文件 | 功能 |
|------|------|
| `lof_detect.py` | LOF 打分（自参照 novelty）、`k_strategy`（k 值动态选取）、`detect`（分位数阈值判定）、`window_features`（滑窗特征） |
| `events.py` | `aggregate_events`（异常点→事件，间隔≤3天合并）、`coincidence`（多参数重合统计） |
| `sensitivity.py` | `quantile_sensitivity`（分位数敏感性）、`k_sensitivity`（k 值敏感性，含 Jaccard 稳定性） |

---

## viz/ — 可视化

| 文件 | 功能 |
|------|------|
| `figures.py` | `plot_prediction`（预测vs观测+残差，对齐文献图3-5）、`plot_anomaly_scores`（异常得分+阈值+事件标注，对齐文献图6-9） |

---

## analysis/ — 分析模块（存根，等数据）

| 文件 | 功能 | 状态 |
|------|------|------|
| `interference.py` | 干扰归因（气象/仪器/人为→tectonic_candidate） | ⏸ 等昭通气象数据 |
| `molchan.py` | Molchan 图表法前兆效能检验 | ⏸ 等地震目录整理 |

---

## 调用关系

```
run_pipeline.py  →  data/parse → data/clean → data/merge → data/split
run_phase2.py    →  features/build → models/{baselines, cnn_lstm, lstm_rf} → models/evaluate → viz/figures
smoke_test.py    →  features/build → models/{baselines, cnn_lstm} → anomaly/{lof_detect, events, sensitivity} → viz/figures
```

所有脚本共享 `src.common.set_seed()` 和 `src.common.load_paths()`。
