// Web 可视化平台 - 主逻辑

const API = '';  // 同源，无需前缀
let currentWindow = 60;
let currentModel = 'cnn_lstm';

// 初始化
async function init() {
    const resp = await fetch(`${API}/api/windows`);
    const data = await resp.json();
    const sel = document.getElementById('windowSelect');
    data.windows.forEach(w => {
        const opt = document.createElement('option');
        opt.value = w; opt.text = w + '天';
        if (w === 60) opt.selected = true;
        sel.appendChild(opt);
    });
    sel.addEventListener('change', e => { currentWindow = +e.target.value; loadAll(); });

    document.getElementById('modelSelect').addEventListener('change', e => {
        currentModel = e.target.value; loadPredictions();
    });
    loadAll();
}

async function loadAll() {
    await Promise.all([loadPredictions(), loadAnomalies(), loadMetrics()]);
}

async function loadPredictions() {
    const resp = await fetch(`${API}/api/predictions/${currentWindow}?model=${currentModel}`);
    const data = await resp.json();
    if (data.error) { console.warn(data.error); return; }
    renderPredictionChart(data);
}

async function loadAnomalies() {
    const resp = await fetch(`${API}/api/anomalies/${currentWindow}`);
    const data = await resp.json();
    renderAnomalyList(data.events || []);
}

async function loadMetrics() {
    const resp = await fetch(`${API}/api/metrics`);
    const data = await resp.json();
    renderMetrics(data);
    renderComparisonChart(data);
}

// 渲染函数
function renderPredictionChart(data) {
    const chart = echarts.init(document.getElementById('predChart'));
    const residual = data.y_true.map((v, i) => +(v - data.y_pred[i]).toFixed(4));
    chart.setOption({
        tooltip: { trigger: 'axis' },
        legend: { data: ['观测值', '预测值', '残差'], top: 10 },
        grid: [{ left: 60, right: 30, top: 50, height: '40%' },
               { left: 60, right: 30, top: '58%', height: '30%' }],
        xAxis: [
            { type: 'category', data: data.dates, gridIndex: 0,
              axisLabel: { formatter: v => v.substring(0, 7) } },
            { type: 'category', data: data.dates, gridIndex: 1,
              axisLabel: { formatter: v => v.substring(0, 7) } }
        ],
        yAxis: [
            { type: 'value', gridIndex: 0, name: '标准化值' },
            { type: 'value', gridIndex: 1, name: '残差' }
        ],
        series: [
            { name: '观测值', type: 'line', data: data.y_true, xAxisIndex: 0, yAxisIndex: 0,
              lineStyle: { width: 1 }, itemStyle: { color: '#999' } },
            { name: '预测值', type: 'line', data: data.y_pred, xAxisIndex: 0, yAxisIndex: 0,
              lineStyle: { width: 1.5 }, itemStyle: { color: '#ff7f0e' } },
            { name: '残差', type: 'line', data: residual, xAxisIndex: 1, yAxisIndex: 1,
              lineStyle: { width: 0.8 }, itemStyle: { color: '#d62728' },
              markLine: { data: [{ yAxis: 0 }], lineStyle: { color: '#333', width: 0.5 } } }
        ]
    });
    window.addEventListener('resize', () => chart.resize());
}

function renderAnomalyList(events) {
    const tbody = document.getElementById('anomalyTable');
    tbody.innerHTML = '';
    events.forEach(e => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${e.event_id || '-'}</td><td>${e.start}~${e.end}</td>`
            + `<td>${e.n_points}</td><td>${(+e.peak_score).toFixed(3)}</td>`;
        tbody.appendChild(tr);
    });
}

function renderMetrics(data) {
    const info = document.getElementById('metricsInfo');
    const cnn = data[currentWindow]?.cnn_lstm;
    const rf = data[currentWindow]?.lstm_rf;
    const ne = data[currentWindow]?.n_events;
    if (!cnn) { info.textContent = '加载中...'; return; }
    info.innerHTML = `<b>CNN-LSTM</b> RMSE=${cnn.RMSE} R²=${cnn.R2} &nbsp;|&nbsp;`
        + `<b>LSTM-RF</b> RMSE=${rf.RMSE} R²=${rf.R2} &nbsp;|&nbsp;`
        + `异常事件: <b>${ne}</b>个 &nbsp;|&nbsp; 窗口: <b>${currentWindow}天</b>`;
}

function renderComparisonChart(data) {
    const chart = echarts.init(document.getElementById('compareChart'));
    const windows = Object.keys(data).sort((a, b) => +a - +b);
    chart.setOption({
        tooltip: { trigger: 'axis' },
        legend: { data: ['CNN-LSTM R²', 'LSTM-RF R²', '异常事件数'] },
        xAxis: { type: 'category', data: windows.map(w => w + '天') },
        yAxis: [
            { type: 'value', name: 'R²', min: 0.5, max: 0.75 },
            { type: 'value', name: '事件数', min: 0, max: 5 }
        ],
        series: [
            { name: 'CNN-LSTM R²', type: 'bar', data: windows.map(w => data[w].cnn_lstm.R2),
              itemStyle: { color: '#ff7f0e' } },
            { name: 'LSTM-RF R²', type: 'bar', data: windows.map(w => data[w].lstm_rf.R2),
              itemStyle: { color: '#1f77b4' } },
            { name: '异常事件数', type: 'bar', yAxisIndex: 1,
              data: windows.map(w => data[w].n_events),
              itemStyle: { color: '#2ca02c' } }
        ]
    });
    window.addEventListener('resize', () => chart.resize());
}

document.addEventListener('DOMContentLoaded', init);
