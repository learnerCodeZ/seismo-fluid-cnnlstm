# Web 可视化平台

基于 **FastAPI + ECharts** 的交互式可视化界面，用于展示项目各阶段的成果。

---

## 一、这是什么

我们项目产出了很多数据和结果（csv、json），但直接看这些文件很不直观。这个 Web 平台把这些成果**变成可交互的图表**，方便：

- **调试**：快速查看数据是否正确、模型预测效果如何
- **展示**：答辩时直接打开浏览器演示，比截图专业
- **汇报**：发一个链接给老师，点开就能看

---

## 二、和项目成果的联系

```
我们项目的成果（数据/模型/异常）
         ↓ 读取
    Web 后端 (FastAPI)
         ↓ 返回 JSON
    Web 前端 (ECharts 图表)
         ↓
      浏览器展示
```

### 具体对应关系

| 我们的成果文件 | Web 展示什么 |
|---------------|-------------|
| `data/clean/long.csv` | 数据概览页的时序曲线（66034条记录，42个台站） |
| `results/phase2_baselines/*/metrics.json` | 模型评估指标表（RMSE/MAE/MAPE/R²） |
| `results/phase2_baselines/*/predictions_test.csv` | 预测对比图（观测值 vs 各模型预测值） |
| `results/phase3_cnn_lstm/cnn_lstm/*` | CNN-LSTM 主模型的结果 |
| `results/smoke/events.csv` | 异常事件列表（3个异常事件） |
| `results/smoke/sensitivity_quantile.csv` | 敏感性分析柱状图 |

---

## 三、如何使用

### 第一步：启动服务

```bash
# 进入 web 目录
cd D:\MYCODE\dachuang_cnn_lstm\web

# 启动服务器
python app.py
```

看到以下信息说明启动成功：
```
Uvicorn running on http://0.0.0.0:8000
```

### 第二步：打开浏览器

访问 **http://localhost:8000**

### 第三步：浏览各页面

#### 页面1：数据概览（主页）
- 左上角选择**台站**（如：昆明基准地震台）
- 选择**测项**（如：水位观测）
- 下方显示该台站的时序曲线
- 可以拖动下方滑块**缩放**查看细节

#### 页面2：模型预测
- 顶部勾选要对比的**模型**（默认全选）
- 上方图表：观测值（黑线）vs 各模型预测值（彩色虚线）
- 下方图表：残差图（预测误差）
- 底部表格：各模型的评估指标

#### 页面3：异常检测
- 调整**阈值分位数**（0.90~0.99），点击"更新"
- 上方图表：异常得分时序，红色点是异常点，黄色虚线是阈值
- 中间表格：异常事件列表
- 下方图表：不同分位数下的异常点数量

---

## 四、常见问题

### Q1: 启动报错 `ModuleNotFoundError: No module named 'uvicorn'`

需要安装依赖：
```bash
pip install fastapi uvicorn -i https://pypi.org/simple/
```

### Q2: 图表不显示 / 数据为空

检查数据文件是否存在：
```
D:\MYCODE\dachuang_cnn_lstm\data\clean\long.csv
D:\MYCODE\dachuang_cnn_lstm\results\phase2_baselines\*
D:\MYCODE\dachuang_cnn_lstm\results\smoke\*
```

### Q3: 想换端口

修改 `app.py` 最后一行：
```python
uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
# 把 8000 改成你想要的端口
```

### Q4: 想在其他电脑上用

把整个 `web/` 文件夹拷贝过去，确保目标电脑有 Python 环境，然后：
```bash
pip install fastapi uvicorn
cd web
python app.py
```

---

## 五、技术细节（可跳过）

### 目录结构

```
web/
├── app.py              # FastAPI 入口，启动服务
├── start_web.bat       # Windows 一键启动脚本
├── api/                # 后端 API
│   ├── data.py         # 读取 long.csv，返回台站/时序数据
│   ├── models.py       # 读取 metrics.json 和 predictions_test.csv
│   └── anomalies.py    # 读取 events.csv 和 sensitivity_quantile.csv
└── static/             # 前端页面
    ├── index.html      # 数据概览页
    ├── prediction.html # 预测对比页
    ├── anomaly.html    # 异常检测页
    ├── css/style.css   # 样式
    └── js/
        ├── data.js         # 数据展示逻辑
        ├── prediction.js   # 预测对比逻辑
        └── anomaly.js      # 异常检测逻辑
```

### API 端点

| 端点 | 说明 |
|------|------|
| `GET /api/stations` | 台站列表 |
| `GET /api/data/{station_id}` | 时序数据 |
| `GET /api/data/{station_id}/stats` | 数据统计 |
| `GET /api/models` | 模型列表 |
| `GET /api/predictions/{model}` | 预测结果 |
| `GET /api/metrics` | 评估指标 |
| `GET /api/anomalies/events` | 异常事件 |
| `GET /api/anomalies/sensitivity` | 敏感性分析 |

### 技术栈

- **后端**：FastAPI（Python）
- **前端**：HTML + JavaScript + ECharts 5（CDN加载）
- **无需**：Node.js、npm、数据库
