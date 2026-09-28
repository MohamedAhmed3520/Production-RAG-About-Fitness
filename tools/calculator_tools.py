from __future__ import annotations

import math
from typing import Dict, List, Optional


class CalculatorTools:
    """Pure calculator utilities that do not rely on an LLM."""

    def bmi(self, weight_kg: float, height_m: float) -> float:
        if height_m <= 0:
            raise ValueError("Height must be greater than zero")
        return round(weight_kg / (height_m * height_m), 2)

    def bmr(self, weight_kg: float, height_cm: float, age: int, sex: str) -> float:
        if sex.lower() == "male":
            result = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            result = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
        return round(result, 2)

    def tdee(self, bmr: float, activity_level: str) -> float:
        multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9,
        }
        return round(bmr * multipliers.get(activity_level.lower(), 1.2), 2)

    def protein(self, weight_kg: float, goal: str = "maintenance") -> float:
        goals = {"fat_loss": 2.2, "maintenance": 1.8, "muscle_gain": 2.2}
        return round(weight_kg * goals.get(goal.lower(), 1.8), 2)

    def carbohydrates(self, calories: float, percent: float = 0.4) -> float:
        return round((calories * percent) / 4, 2)

    def fat(self, calories: float, percent: float = 0.3) -> float:
        return round((calories * percent) / 9, 2)

    def water_intake(self, weight_kg: float, activity_level: str = "moderate") -> float:
        multiplier = {"sedentary": 30, "light": 35, "moderate": 35, "active": 40, "very_active": 45}.get(activity_level.lower(), 35)
        return round(weight_kg * multiplier / 1000, 2)

    def calorie_deficit(self, maintenance_calories: float, deficit_percent: float = 0.2) -> float:
        return round(maintenance_calories * (1 - deficit_percent), 2)

    def calorie_surplus(self, maintenance_calories: float, surplus_percent: float = 0.1) -> float:
        return round(maintenance_calories * (1 + surplus_percent), 2)

    def one_rep_max(self, weight_kg: float, reps: int) -> float:
        if reps <= 0:
            raise ValueError("Reps must be greater than zero")
        return round(weight_kg * (1 + reps / 30), 2)
