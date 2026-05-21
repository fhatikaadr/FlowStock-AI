from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class PolicyConfig:
    lead_time_days: int = 4
    service_level: float = 0.95
    overstock_factor: float = 1.35


def z_value(service_level: float) -> float:
    return float(norm.ppf(service_level))


def safety_stock(std_daily_demand: float, lead_time_days: int, service_level: float) -> float:
    if lead_time_days <= 0:
        return 0.0
    return z_value(service_level) * (std_daily_demand * math.sqrt(lead_time_days))


def reorder_point(avg_daily_demand: float, std_daily_demand: float, cfg: PolicyConfig) -> float:
    return (avg_daily_demand * cfg.lead_time_days) + safety_stock(
        std_daily_demand=std_daily_demand,
        lead_time_days=cfg.lead_time_days,
        service_level=cfg.service_level,
    )


def classify_status(current_stock: float, demand_14d: float, reorder_pt: float, cfg: PolicyConfig) -> str:
    if current_stock < reorder_pt or current_stock < demand_14d:
        return "Critical"
    if current_stock > demand_14d * cfg.overstock_factor:
        return "Overstock"
    return "Healthy"
