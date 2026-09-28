from app.tools.calculator_tools import (
    bmi_calculator,
    bmr_calculator,
    carb_calculator,
    fat_calculator,
    one_rep_max_calculator,
    protein_calculator,
    tdee_calculator,
    water_intake_calculator,
)


def test_bmi_calculator() -> None:
    assert bmi_calculator.run(70, 1.75) == 22.86


def test_bmr_calculator() -> None:
    assert bmr_calculator.run(70, 175, 30, "male") == 1600.0


def test_protein_calculator() -> None:
    assert protein_calculator.run(70, "maintenance") == 126.0


def test_carb_calculator() -> None:
    assert carb_calculator.run(2000, 0.4) == 200.0


def test_fat_calculator() -> None:
    assert fat_calculator.run(2000, 0.3) == 66.67


def test_water_intake_calculator() -> None:
    assert water_intake_calculator.run(70, "moderate") == 2.45


def test_tdee_calculator() -> None:
    assert tdee_calculator.run(1600, "moderate") == 2480.0


def test_one_rep_max_calculator() -> None:
    assert one_rep_max_calculator.run(100, 8) == 126.67
