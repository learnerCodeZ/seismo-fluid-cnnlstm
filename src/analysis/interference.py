"""analysis/interference.py — 干扰排除（存根：等待气象三要素数据到位后实现）。

输入约定（届时由数据管道补充）：
- meteo: DataFrame(date, precip, temp, pressure) 日值，已标准化说明见《实施指南》
- events: 阶段4/5 的异常事件表（见 src/anomaly/events.py）

输出约定：
- verdicts: DataFrame(event_id, attribution, evidence)——attribution ∈
  {rain_dilution, temp_effect, pressure_pumping, instrument, anthropogenic, tectonic_candidate}
"""
from __future__ import annotations


def attribute_events(events, meteo=None, calibration_log=None, anthropogenic=None):
    """对每个候选异常事件做干扰归因。

    规则（实现时依据文献 §3.3）：
    1. 事件时段与强降水（日雨量超阈值）重叠且水氡为低值 → rain_dilution；
    2. 与高温时段重叠且为高值 → temp_effect（脱气增强）；
    3. 与气压骤变相关且相位一致 → pressure_pumping；
    4. 与标定/检修记录日期邻近 → instrument；
    5. 与人为活动记录邻近 → anthropogenic；
    6. 以上均不能解释 → tectonic_candidate（进入震例对应分析）。
    """
    raise NotImplementedError("等待气象三要素数据（见 docs/数据需求清单.md 2.1）")
