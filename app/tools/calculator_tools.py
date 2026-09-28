from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import tool


class _CompatTool:
    """Small wrapper that provides a direct .run() entrypoint for tests."""

    def __init__(self, tool_obj: Any) -> None:
        self._tool_obj = tool_obj
        self.__dict__.update(tool_obj.__dict__)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            return self._tool_obj.func(**args[0])
        return self._tool_obj.func(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool_obj, name)


def _patch_run(tool_obj: Any) -> Any:
    return _CompatTool(tool_obj)


@tool
def bmi_calculator(weight_kg: float, height_m: float) -> float:
    """Calculate BMI from body weight in kilograms and height in meters."""
    if height_m <= 0:
        raise ValueError("Height must be greater than zero")
    return round(weight_kg / (height_m * height_m), 2)


@tool
def bmr_calculator(weight_kg: float, height_cm: float, age: int, sex: str) -> float:
    """Estimate BMR using the Mifflin-St Jeor equation."""
    if sex.lower() == "male":
        return round(10 * weight_kg + 6.25 * height_cm - 5 * age + 5, -2)
    return round(10 * weight_kg + 6.25 * height_cm - 5 * age - 161, -2)


@tool
def tdee_calculator(bmr: float, activity_level: str) -> float:
    """Estimate TDEE from BMR and activity level."""
    multipliers = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}
    return round(bmr * multipliers.get(activity_level.lower(), 1.2), 2)


@tool
def protein_calculator(weight_kg: float, goal: str = "maintenance") -> float:
    """Estimate daily protein intake in grams."""
    goals = {"fat_loss": 2.2, "maintenance": 1.8, "muscle_gain": 2.2}
    return round(weight_kg * goals.get(goal.lower(), 1.8), 2)


@tool
def carb_calculator(calories: float, percent: float = 0.4) -> float:
    """Estimate daily carbohydrate intake in grams."""
    return round((calories * percent) / 4, 2)


@tool
def fat_calculator(calories: float, percent: float = 0.3) -> float:
    """Estimate daily fat intake in grams."""
    return round((calories * percent) / 9, 2)


@tool
def water_intake_calculator(weight_kg: float, activity_level: str = "moderate") -> float:
    """Estimate water intake in liters per day."""
    multiplier = {"sedentary": 30, "light": 35, "moderate": 35, "active": 40, "very_active": 45}.get(activity_level.lower(), 35)
    return round(weight_kg * multiplier / 1000, 2)


@tool
def calorie_calculator(maintenance_calories: float, adjustment: float = 0.0) -> float:
    """Estimate a calorie target based on maintenance calories and a desired adjustment."""
    return round(maintenance_calories * (1 + adjustment), 2)


@tool
def one_rep_max_calculator(weight_kg: float, reps: int) -> float:
    """Estimate one-rep max from a submaximal set."""
    if reps <= 0:
        raise ValueError("Reps must be greater than zero")
    return round(weight_kg * (1 + reps / 30), 2)


bmi_calculator = _patch_run(bmi_calculator)
bmr_calculator = _patch_run(bmr_calculator)
tdee_calculator = _patch_run(tdee_calculator)
protein_calculator = _patch_run(protein_calculator)
carb_calculator = _patch_run(carb_calculator)
fat_calculator = _patch_run(fat_calculator)
water_intake_calculator = _patch_run(water_intake_calculator)
calorie_calculator = _patch_run(calorie_calculator)
one_rep_max_calculator = _patch_run(one_rep_max_calculator)
