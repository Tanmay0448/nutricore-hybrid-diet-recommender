"""Nutrition target calculations used by the NutriCore application."""

import pandas as pd


_ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.20,
    "lightly active": 1.375,
    "moderately active": 1.55,
    "very active": 1.725,
    "athlete": 1.90,
}

_PROTEIN_TARGETS = {
    "Maintenance": 1.6,
    "Mild Cut": 2.0,
    "Moderate Cut": 2.2,
    "Aggressive Cut": 2.4,
    "Lean Bulk": 2.0,
    "Bulk": 2.0,
    "Aggressive Bulk": 2.0,
}


def calculate_bmr(age: int, gender: str, weight: float, height: float) -> float:
    """Calculate BMR with the Mifflin-St Jeor equation."""
    adjustment = {"male": 5, "female": -161}.get(gender.lower())
    if adjustment is None:
        raise ValueError("Sex must be either 'male' or 'female'.")
    return (10 * weight) + (6.25 * height) - (5 * age) + adjustment


def calculate_tdee(bmr: float, activity_level: str) -> float:
    """Apply an activity multiplier to a BMR value."""
    try:
        return bmr * _ACTIVITY_MULTIPLIERS[activity_level.lower()]
    except KeyError as error:
        raise ValueError("Invalid activity level provided.") from error


def generate_calorie_targets(tdee: float) -> dict[str, int]:
    """Return calorie recommendations for each supported goal."""
    multipliers = {
        "Maintenance": 1.00,
        "Mild Cut": 0.90,
        "Moderate Cut": 0.80,
        "Aggressive Cut": 0.75,
        "Lean Bulk": 1.05,
        "Bulk": 1.10,
        "Aggressive Bulk": 1.15,
    }
    return {goal: round(tdee * multiplier) for goal, multiplier in multipliers.items()}


def _macro_targets(weight: float, calories: float, goal: str) -> dict[str, float]:
    protein = round(weight * _PROTEIN_TARGETS[goal], 1)
    fat = round((calories * 0.25) / 9, 1)
    carbs = round((calories - (protein * 4) - (fat * 9)) / 4, 1)
    return {"Calories": round(calories), "Protein_g": protein, "Fat_g": fat, "Carbs_g": carbs}


def generate_all_goal_profiles(weight: float, calorie_targets: dict[str, int]) -> pd.DataFrame:
    """Build the target cards displayed for each nutrition goal."""
    profiles = []
    for goal, calories in calorie_targets.items():
        macros = _macro_targets(weight, calories, goal)
        profiles.append({"Goal": goal, **macros})
    return pd.DataFrame(profiles)


def generate_manual_targets(
    calories: float,
    protein: float,
    fat: float | None = None,
    carbs: float | None = None,
) -> dict[str, float]:
    """Complete missing manual macros while preserving supplied values."""
    if fat is None and carbs is None:
        fat = round((calories * 0.25) / 9, 1)
        carbs = round((calories - (protein * 4) - (fat * 9)) / 4, 1)
    elif carbs is None:
        carbs = round((calories - (protein * 4) - (fat * 9)) / 4, 1)
    elif fat is None:
        fat = round((calories - (protein * 4) - (carbs * 4)) / 9, 1)
    return {"Calories": calories, "Protein_g": protein, "Fat_g": fat, "Carbs_g": carbs}


def validate_macros(
    calories: float,
    protein: float,
    fat: float,
    carbs: float,
    tolerance: float = 50,
) -> dict[str, float | bool]:
    """Check whether the supplied macro calories agree with the calorie target."""
    macro_calories = (protein * 4) + (fat * 9) + (carbs * 4)
    difference = round(abs(calories - macro_calories), 1)
    return {
        "Target Calories": calories,
        "Macro Calories": round(macro_calories, 1),
        "Difference": difference,
        "Valid": difference <= tolerance,
    }


def assess_macro_quality(calories: float, protein: float, fat: float, carbs: float) -> dict[str, float | list[str]]:
    """Return macro percentages and non-blocking quality warnings."""
    protein_pct = (protein * 4 / calories) * 100
    fat_pct = (fat * 9 / calories) * 100
    carb_pct = (carbs * 4 / calories) * 100
    warnings = []
    recommendations = []
    if protein_pct < 15:
        recommended_minimum = round((calories * 0.15) / 4)
        warnings.append("Protein intake is below the recommended range for active individuals.")
        recommendations.append(f"Protein: {protein:.0f} g selected; aim for at least {recommended_minimum:.0f} g (about 15% of calories).")
    elif protein_pct > 40:
        warnings.append("Protein intake appears unusually high.")
        recommendations.append(f"Protein: {protein:.0f} g selected; consider staying below {round((calories * 0.40) / 4):.0f} g (about 40% of calories).")
    if fat_pct < 20:
        warnings.append("Fat intake appears lower than recommended.")
        recommendations.append(f"Fat: {fat:.0f} g selected; aim for at least {round((calories * 0.20) / 9):.0f} g (about 20% of calories).")
    elif fat_pct > 35:
        warnings.append("Fat intake appears higher than recommended.")
        recommendations.append(f"Fat: {fat:.0f} g selected; consider staying below {round((calories * 0.35) / 9):.0f} g (about 35% of calories).")
    if carb_pct < 20:
        warnings.append("Carbohydrate intake appears lower than recommended.")
        recommendations.append(f"Carbs: {carbs:.0f} g selected; aim for at least {round((calories * 0.20) / 4):.0f} g (about 20% of calories).")
    return {
        "Protein_%": round(protein_pct, 1),
        "Fat_%": round(fat_pct, 1),
        "Carbohydrates_%": round(carb_pct, 1),
        "Warnings": warnings or ["No Warnings"],
        "Recommendations": recommendations,
    }


def check_recommendation_eligibility(calories: float, protein: float, fat: float, carbs: float) -> dict[str, bool | list[str]]:
    """Reject macro targets that are too low to safely recommend against."""
    percentages = {"Protein": protein * 4 / calories * 100, "Fat": fat * 9 / calories * 100, "Carbohydrate": carbs * 4 / calories * 100}
    reasons = []
    if percentages["Protein"] < 10:
        reasons.append(f"Protein is too low for recommendations: use at least {round((calories * 0.10) / 4):.0f} g (10% of calories).")
    if percentages["Fat"] < 10:
        reasons.append(f"Fat is too low for recommendations: use at least {round((calories * 0.10) / 9):.0f} g (10% of calories).")
    if percentages["Carbohydrate"] < 5:
        reasons.append(f"Carbs are too low for recommendations: use at least {round((calories * 0.05) / 4):.0f} g (5% of calories).")
    return {"Eligible": not reasons, "Reasons": reasons or ["No Issues"]}


__all__ = [
    "assess_macro_quality",
    "calculate_bmr",
    "calculate_tdee",
    "check_recommendation_eligibility",
    "generate_all_goal_profiles",
    "generate_calorie_targets",
    "generate_manual_targets",
    "validate_macros",
]
