let scoresChart = null;
let sensitivityChart = null;

async function init() {
    await Promise.all([
        loadSummary(),
        loadScores(),
        loadEvents(),
        loadSensitivity(),
    ]);
}

async function loadSummary() {
    const res = await fetch('/api/anomalies/summary');
    const data = await res.json();

    document.getElementById('stat-events').textContent = data.total_events;
    document.getElementById('stat-verified').textContent = data.verified_events;
    document.getElementById('stat-m4').textContent = data.total_m4_eq;
    document.getElementById('stat-m5').textContent = data.total_m5_eq;
}

async function loadScores() {
    const res = await fetch('/api/anomalies/scores?quantile=0.98');
    const data = await res.json();
    renderScoresChart(data);
}

function renderScoresChart(data) {
    if (!scoresChart) {
        scoresChart = echarts.init(document.getElementById('scores-chart'));
        window.addEventListener('resize', () => scoresChart.resize());
    }

    const validDates = data.dates.filter(d => d !== null);
    const validScores = data.scores.filter((s, i) => data.dates[i] !== null);

    const anomalyData = [];
    validDates.forEach((d, i) => {
        if (validScores[i] >= data.threshold) {
            anomalyData.push([d, validScores[i]]);
        }
    });

    const option = {
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                const point = params[0];
                if (!point) return '';
                return `${point.axisValue}<br/>残差得分: ${point.value[1]}`;
            }
        },
        legend: {
            data: ['残差得分', '异常点', '阈值线'],
            top: 10,
        },
        grid: { left: 60, right: 30, top: 50, bottom: 40 },
        xAxis: {
            type: 'category',
            data: validDates,
            axisLabel: {
                formatter: function(val) {
                    return val ? val.substring(0, 10) : '';
                }
            },
            boundaryGap: false,
        },
        yAxis: {
            type: 'value',
            name: '残差得分',
        },
        dataZoom: [
            { type: 'inside', start: 0, end: 100 },
            { type: 'slider', start: 0, end: 100, height: 20, bottom: 5 }
        ],
        series: [
            {
                name: '残差得分',
                type: 'line',
                data: validScores,
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
                data: validDates.map(() => data.threshold),
                symbol: 'none',
                lineStyle: { width: 1, color: '#fac858', type: 'dashed' },
                itemStyle: { color: '#fac858' },
            }
        ]
    };

    scoresChart.setOption(option, true);
}

async function loadEvents() {
    const res = await fetch('/api/anomalies/events');
    const events = await res.json();

    const tbody = document.getElementById('events-body');
    tbody.innerHTML = events.map(e => {
        const isVerified = e.correspondence === '✓';
        return `
            <tr class="${isVerified ? 'verified' : ''}">
                <td>${e.id}</td>
                <td>${e.start} ~ ${e.end}</td>
                <td>${e.n_points}</td>
                <td>${e.peak_score}</td>
                <td>${e.direction}</td>
                <td>${e.precip_mm} mm</td>
                <td><span class="tag tag-${e.attribution.includes('气象') ? 'warning' : 'success'}">${e.attribution}</span></td>
                <td>${e.eq_m4}</td>
                <td>${e.eq_m5}</td>
                <td><span class="tag tag-${isVerified ? 'success' : 'info'}">${e.correspondence}</span></td>
            </tr>
        `;
    }).join('');

    renderEarthquakeDetails(events);
}

function renderEarthquakeDetails(events) {
    const container = document.getElementById('earthquake-details');
    const withEq = events.filter(e => e.eq_m5 > 0 && e.eq_m5_details !== '无');

    if (withEq.length === 0) {
        container.innerHTML = '<p class="no-data">暂无M5+地震对应数据</p>';
        return;
    }

    let html = '<div class="eq-list">';
    withEq.forEach(e => {
        const details = e.eq_m5_details.split(';').filter(d => d.trim());
        html += `
            <div class="eq-item">
                <div class="eq-header">
                    <span class="eq-badge">事件 #${e.id}</span>
                    <span class="eq-date">${e.start} ~ ${e.end}</span>
                    <span class="eq-count">${e.eq_m5} 个M5+地震</span>
                </div>
                <ul class="eq-details">
                    ${details.map(d => `<li>${d.trim()}</li>`).join('')}
                </ul>
            </div>
        `;
    });
    html += '</div>';
    container.innerHTML = html;
}

async function loadSensitivity() {
    const res = await fetch('/api/anomalies/sensitivity');
    const data = await res.json();
    renderSensitivityChart(data);
}

function renderSensitivityChart(data) {
    if (!sensitivityChart) {
        sensitivityChart = echarts.init(document.getElementById('sensitivity-chart'));
        window.addEventListener('resize', () => sensitivityChart.resize());
    }

    const option = {
        tooltip: { trigger: 'axis' },
        grid: { left: 60, right: 30, top: 30, bottom: 40 },
        xAxis: {
            type: 'category',
            data: data.map(d => d.quantile),
            axisLabel: { formatter: '{value}' }
        },
        yAxis: {
            type: 'value',
            name: '异常事件数',
        },
        series: [{
            name: '异常事件数',
            type: 'bar',
            data: data.map(d => d.n_anomalies),
            itemStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: '#667eea' },
                    { offset: 1, color: '#764ba2' }
                ])
            },
            label: {
                show: true,
                position: 'top',
                formatter: '{c}'
            }
        }]
    };

    sensitivityChart.setOption(option, true);
}

init();
