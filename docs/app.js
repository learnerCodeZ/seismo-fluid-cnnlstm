let appData = null;

async function init() {
    try {
        const response = await fetch('data.json');
        appData = await response.json();
        renderDataTab();
        renderPredictionTab();
        renderAnomalyTab();
    } catch (error) {
        console.error('加载数据失败:', error);
    }
}

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.nav-links a').forEach(link => link.classList.remove('active'));

    document.getElementById(tabName + '-tab').classList.add('active');
    document.querySelector(`.nav-links a[href="#${tabName}"]`).classList.add('active');
}

function renderDataTab() {
    if (!appData || !appData.stations) return;

    const station = appData.stations[0];
    document.getElementById('station-name').textContent = station.name;
    document.getElementById('stat-days').textContent = station.n_days.toLocaleString() + ' 条';
    document.getElementById('date-range').textContent = station.date_range;

    const chart = echarts.init(document.getElementById('timeseries-chart'));
    const predictions = appData.predictions;

    const option = {
        tooltip: { trigger: 'axis' },
        grid: { left: 60, right: 30, top: 50, bottom: 40 },
        xAxis: {
            type: 'category',
            data: predictions.dates,
            axisLabel: {
                formatter: function(val) {
                    return val ? val.substring(0, 10) : '';
                }
            },
            boundaryGap: false,
        },
        yAxis: { type: 'value', name: '观测值' },
        dataZoom: [
            { type: 'inside', start: 0, end: 100 },
            { type: 'slider', start: 0, end: 100, height: 20, bottom: 5 }
        ],
        series: [{
            name: '观测值',
            type: 'line',
            data: predictions.y_true,
            symbol: 'none',
            lineStyle: { width: 1.5, color: '#667eea' },
            itemStyle: { color: '#667eea' },
        }]
    };

    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
}

function renderPredictionTab() {
    if (!appData) return;

    const predictions = appData.predictions;
    const metrics = appData.metrics;
    const MODEL_COLORS = {
        naive: '#999999',
        rf: '#5470c6',
        lstm: '#91cc75',
        lstm_rf: '#fac858',
        cnn_lstm: '#ee6666',
    };
    const MODEL_NAMES = {
        naive: 'Naive',
        rf: '随机森林',
        lstm: 'LSTM',
        lstm_rf: 'LSTM-RF',
        cnn_lstm: 'CNN-LSTM',
    };

    // 预测图表
    const predChart = echarts.init(document.getElementById('prediction-chart'));
    const predSeries = [
        {
            name: '观测值',
            type: 'line',
            data: predictions.y_true,
            symbol: 'none',
            lineStyle: { width: 2, color: '#333' },
            itemStyle: { color: '#333' },
            z: 10,
        }
    ];

    Object.keys(predictions.models).forEach(model => {
        const m = predictions.models[model];
        predSeries.push({
            name: m.display,
            type: 'line',
            data: m.y_pred,
            symbol: 'none',
            lineStyle: { width: 1.5, color: m.color, type: 'dashed' },
            itemStyle: { color: m.color },
        });
    });

    predChart.setOption({
        tooltip: { trigger: 'axis' },
        legend: { data: ['观测值', ...Object.values(predictions.models).map(m => m.display)], top: 10 },
        grid: { left: 60, right: 30, top: 50, bottom: 40 },
        xAxis: {
            type: 'category',
            data: predictions.dates,
            axisLabel: { formatter: val => val ? val.substring(0, 10) : '' },
            boundaryGap: false,
        },
        yAxis: { type: 'value', name: '观测值' },
        dataZoom: [
            { type: 'inside', start: 0, end: 100 },
            { type: 'slider', start: 0, end: 100, height: 20, bottom: 5 }
        ],
        series: predSeries,
    });

    // 残差图
    const residualChart = echarts.init(document.getElementById('residual-chart'));
    const residualSeries = [];

    Object.keys(predictions.models).forEach(model => {
        const m = predictions.models[model];
        const residual = predictions.y_true.map((y, i) => y - m.y_pred[i]);
        residualSeries.push({
            name: m.display,
            type: 'line',
            data: residual,
            symbol: 'none',
            lineStyle: { width: 1, color: m.color },
            itemStyle: { color: m.color },
        });
    });

    residualSeries.push({
        name: '零线',
        type: 'line',
        data: predictions.dates.map(() => 0),
        symbol: 'none',
        lineStyle: { width: 1, color: '#ccc', type: 'dashed' },
        itemStyle: { color: '#ccc' },
    });

    residualChart.setOption({
        tooltip: { trigger: 'axis' },
        legend: { data: [...Object.values(predictions.models).map(m => m.display), '零线'], top: 10 },
        grid: { left: 60, right: 30, top: 50, bottom: 40 },
        xAxis: {
            type: 'category',
            data: predictions.dates,
            axisLabel: { formatter: val => val ? val.substring(0, 10) : '' },
            boundaryGap: false,
        },
        yAxis: { type: 'value', name: '残差' },
        dataZoom: [
            { type: 'inside', start: 0, end: 100 },
            { type: 'slider', start: 0, end: 100, height: 20, bottom: 5 }
        ],
        series: residualSeries,
    });

    // 指标表
    const tbody = document.getElementById('metrics-body');
    tbody.innerHTML = Object.entries(metrics).map(([name, m]) => `
        <tr>
            <td><span style="color:${MODEL_COLORS[name]}">&#9679;</span> ${MODEL_NAMES[name]}</td>
            <td>${m.RMSE.toFixed(4)}</td>
            <td>${m.MAE.toFixed(4)}</td>
            <td>${m.MAPE.toFixed(2)}%</td>
            <td>${m.R2.toFixed(4)}</td>
        </tr>
    `).join('');

    window.addEventListener('resize', () => {
        predChart.resize();
        residualChart.resize();
    });
}

function renderAnomalyTab() {
    if (!appData) return;

    const events = appData.events;
    const sensitivity = appData.sensitivity;

    // 统计
    document.getElementById('stat-events').textContent = events.length;
    document.getElementById('stat-verified').textContent = events.filter(e => e.correspondence === '✓').length;
    document.getElementById('stat-m4').textContent = events.reduce((sum, e) => sum + e.eq_m4, 0);

    // 得分图表
    const scoresChart = echarts.init(document.getElementById('scores-chart'));
    const predictions = appData.predictions;
    const residual = predictions.y_true.map((y, i) => Math.abs(y - predictions.models.cnn_lstm.y_pred[i]));
    const threshold = residual.sort((a, b) => a - b)[Math.floor(residual.length * 0.98)];

    const anomalyData = [];
    predictions.dates.forEach((d, i) => {
        if (residual[i] >= threshold) {
            anomalyData.push([d, residual[i]]);
        }
    });

    scoresChart.setOption({
        tooltip: { trigger: 'axis' },
        legend: { data: ['残差得分', '异常点', '阈值线'], top: 10 },
        grid: { left: 60, right: 30, top: 50, bottom: 40 },
        xAxis: {
            type: 'category',
            data: predictions.dates,
            axisLabel: { formatter: val => val ? val.substring(0, 10) : '' },
            boundaryGap: false,
        },
        yAxis: { type: 'value', name: '残差得分' },
        dataZoom: [
            { type: 'inside', start: 0, end: 100 },
            { type: 'slider', start: 0, end: 100, height: 20, bottom: 5 }
        ],
        series: [
            {
                name: '残差得分',
                type: 'line',
                data: residual,
                symbol: 'none',
                lineStyle: { width: 1.5, color: '#5470c6' },
                itemStyle: { color: '#5470c6' },
            },
            {
                name: '异常点',
                type: 'scatter',
                data: anomalyData,
                symbolSize: 10,
                itemStyle: { color: '#ee6666' },
            },
            {
                name: '阈值线',
                type: 'line',
                data: predictions.dates.map(() => threshold),
                symbol: 'none',
                lineStyle: { width: 1, color: '#fac858', type: 'dashed' },
                itemStyle: { color: '#fac858' },
            }
        ]
    });

    // 事件表
    const eventsBody = document.getElementById('events-body');
    eventsBody.innerHTML = events.map(e => `
        <tr class="${e.correspondence === '✓' ? 'verified' : ''}">
            <td>${e.id}</td>
            <td>${e.start} ~ ${e.end}</td>
            <td>${e.n_points}</td>
            <td>${e.peak_score}</td>
            <td><span class="tag tag-${e.attribution.includes('气象') ? 'warning' : 'success'}">${e.attribution}</span></td>
            <td>${e.eq_m4}</td>
            <td><span class="tag tag-${e.correspondence === '✓' ? 'success' : 'info'}">${e.correspondence}</span></td>
        </tr>
    `).join('');

    // 敏感性图表
    const sensitivityChart = echarts.init(document.getElementById('sensitivity-chart'));
    sensitivityChart.setOption({
        tooltip: { trigger: 'axis' },
        grid: { left: 60, right: 30, top: 30, bottom: 40 },
        xAxis: {
            type: 'category',
            data: sensitivity.map(d => d.quantile),
            axisLabel: { formatter: '{value}' }
        },
        yAxis: { type: 'value', name: '异常事件数' },
        series: [{
            name: '异常事件数',
            type: 'bar',
            data: sensitivity.map(d => d.n_anomalies),
            itemStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: '#667eea' },
                    { offset: 1, color: '#764ba2' }
                ])
            },
            label: { show: true, position: 'top', formatter: '{c}' }
        }]
    });

    window.addEventListener('resize', () => {
        scoresChart.resize();
        sensitivityChart.resize();
    });
}

init();
