"""analysis/molchan.py — 前兆预测效能检验（Molchan 图表法，存根）。

用途：《研究计划书》风险预案——震例少时用全目录统计检验异常-地震对应的显著性。
参照：鲁明贵等（2024）基于 Molchan 图表法的流体监测井水位地震预测效能检验。

输入约定：
- anomaly_events: 异常事件表（start, end）
- catalog: DataFrame(date, mag, dist_km)——研究期地震目录
- config: 告警窗口（异常事件后 N 天）、震级下限、震中距上限
输出约定：miss/ hit 序列 → Molchan 轨迹曲线 + 眼镜猴检验显著性面积。
"""
from __future__ import annotations


def molchan_test(anomaly_events, catalog, alarm_days: int = 90,
                 min_mag: float = 4.0, max_dist_km: float = 400.0):
    raise NotImplementedError("等待地震目录整理（见 docs/数据需求清单.md 3.1）")
