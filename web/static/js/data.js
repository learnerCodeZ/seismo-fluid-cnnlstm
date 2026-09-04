let chart = null;

async function init() {
    await loadStations();
    document.getElementById('station-select').addEventListener('change', onStationChange);
    document.getElementById('item-select').addEventListener('change', loadData);
    if (document.getElementById('station-select').value) {
        onStationChange();
    }
}

async function loadStations() {
    const res = await fetch('/api/stations');
    const stations = await res.json();
    const select = document.getElementById('station-select');
    select.innerHTML = stations.map(s =>
        `<option value="${s.id}">${s.name}</option>`
    ).join('');

    if (stations.length > 0) {
        updateItems(stations[0].items);
    }
}

function updateItems(items) {
    const select = document.getElementById('item-select');
    select.innerHTML = items.map(item =>
        `<option value="${item}">${item}</option>`
    ).join('');
}

async function onStationChange() {
    const stationId = document.getElementById('station-select').value;
    if (!stationId) return;

    const res = await fetch('/api/stations');
    const stations = await res.json();
    const station = stations.find(s => s.id === stationId);

    if (station && station.items) {
        updateItems(station.items);
    }

    loadData();
}

async function loadData() {
    const stationId = document.getElementById('station-select').value;
    const item = document.getElementById('item-select').value;
    if (!stationId) return;

    const [dataRes, statsRes] = await Promise.all([
        fetch(`/api/data/${stationId}?item=${encodeURIComponent(item)}`),
        fetch(`/api/data/${stationId}/stats?item=${encodeURIComponent(item)}`),
    ]);

    const data = await dataRes.json();
    const stats = await statsRes.json();

    document.getElementById('stat-days').textContent = stats.n_days.toLocaleString();
    document.getElementById('stat-missing').textContent = (stats.missing_ratio * 100).toFixed(2) + '%';
    document.getElementById('stat-quality').textContent = stats.quality === 'usable' ? '可用' : '部分缺失';
    document.getElementById('date-range').textContent = stats.date_range || '-';

    renderChart(data.dates, data.values, item);
}

function renderChart(dates, values, item) {
    if (!chart) {
        chart = echarts.init(document.getElementById('timeseries-chart'));
        window.addEventListener('resize', () => chart.resize());
    }

    const validData = [];
    const missingData = [];

    for (let i = 0; i < dates.length; i++) {
        const date = dates[i];
        const val = values[i];
        if (val !== null && val !== undefined) {
            validData.push([date, val]);
        } else {
            missingData.push([date, 0]);
        }
    }

    const option = {
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                const point = params[0];
                if (!point) return '';
                return `${point.axisValue}<br/>${item}: ${point.value[1]}`;
            }
        },
        legend: {
            data: [item, '缺测'],
            top: 10,
        },
        grid: {
            left: 60,
            right: 30,
            top: 50,
            bottom: 40,
        },
        xAxis: {
            type: 'category',
            data: dates,
            axisLabel: {
                formatter: function(val) {
                    return val ? val.substring(0, 7) : '';
                }
            },
            boundaryGap: false,
        },
        yAxis: {
            type: 'value',
            name: item,
        },
        dataZoom: [
            { type: 'inside', start: 0, end: 100 },
            { type: 'slider', start: 0, end: 100, height: 20, bottom: 5 }
        ],
        series: [
            {
                name: item,
                type: 'line',
                data: validData.map(d => d[1]),
                symbol: 'none',
                lineStyle: { width: 1.5, color: '#667eea' },
                itemStyle: { color: '#667eea' },
                connectNulls: false,
            },
            {
                name: '缺测',
                type: 'scatter',
                data: missingData,
                symbol: 'triangle',
                symbolSize: 8,
                itemStyle: { color: '#ee6666' },
            }
        ]
    };

    chart.setOption(option, true);
}

init();
