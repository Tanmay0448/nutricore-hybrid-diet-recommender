# =====================================================
# Nutrition Module
# =====================================================

import pandas as pd

# =====================================================
# Activity Multipliers
# =====================================================

ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.20,
    "lightly active": 1.375,
    "moderately active": 1.55,
    "very active": 1.725,
    "athlete": 1.90
}

# =====================================================
# Protein Targets (g/kg Body Weight)
# =====================================================

PROTEIN_TARGETS = {
    "Maintenance": 1.6,
    "Mild Cut": 2.0,
    "Moderate Cut": 2.2,
    "Aggressive Cut": 2.4,
    "Lean Bulk": 2.0,
    "Bulk": 2.0,
    "Aggressive Bulk": 2.0
}


# =====================================================
# BMR Calculation
# =====================================================

def calculate_bmr(age, gender, weight, height):

    gender = gender.lower()

    if gender == "male":
        return (10 * weight) + (6.25 * height) - (5 * age) + 5

    elif gender == "female":
        return (10 * weight) + (6.25 * height) - (5 * age) - 161

    raise ValueError(
        "Gender must be either 'male' or 'female'"
    )


# =====================================================
# TDEE Calculation
# =====================================================

def calculate_tdee(bmr, activity_level):

    activity_level = activity_level.lower()

    if activity_level not in ACTIVITY_MULTIPLIERS:
        raise ValueError(
            "Invalid activity level provided."
        )

    return (
        bmr *
        ACTIVITY_MULTIPLIERS[activity_level]
    )


# =====================================================
# Goal Based Calorie Targets
# =====================================================

def generate_calorie_targets(tdee):

    return {
        "Maintenance": round(tdee),

        "Mild Cut": round(tdee * 0.90),

        "Moderate Cut": round(tdee * 0.80),

        "Aggressive Cut": round(tdee * 0.75),

        "Lean Bulk": round(tdee * 1.05),

        "Bulk": round(tdee * 1.10),

        "Aggressive Bulk": round(tdee * 1.15)
    }


# =====================================================
# Protein Calculation
# =====================================================

def calculate_protein(weight, goal):

    return round(
        weight *
        PROTEIN_TARGETS[goal],
        1
    )


# =====================================================
# Fat Calculation
# =====================================================

def calculate_fat(calories):

    fat_calories = calories * 0.25

    return round(
        fat_calories / 9,
        1
    )


# =====================================================
# Carbohydrate Calculation
# =====================================================

def calculate_carbs(
    calories,
    protein_grams,
    fat_grams
):

    protein_calories = protein_grams * 4

    fat_calories = fat_grams * 9

    remaining_calories = (
        calories
        - protein_calories
        - fat_calories
    )

    return round(
        remaining_calories / 4,
        1
    )


# =====================================================
# Macro Target Generator
# =====================================================

def generate_macro_targets(
    weight,
    calories,
    goal
):

    protein = calculate_protein(
        weight,
        goal
    )

    fat = calculate_fat(
        calories
    )

    carbs = calculate_carbs(
        calories,
        protein,
        fat
    )

    return {
        "Calories": round(calories),
        "Protein_g": protein,
        "Fat_g": fat,
        "Carbs_g": carbs
    }


# =====================================================
# Nutrition Profiles For All Goals
# =====================================================

def generate_all_goal_profiles(
    weight,
    calorie_targets
):

    profiles = []

    for goal, calories in calorie_targets.items():

        macros = generate_macro_targets(
            weight=weight,
            calories=calories,
            goal=goal
        )

        profiles.append({
            "Goal": goal,
            "Calories": macros["Calories"],
            "Protein_g": macros["Protein_g"],
            "Fat_g": macros["Fat_g"],
            "Carbs_g": macros["Carbs_g"]
        })

    return pd.DataFrame(profiles)


# =====================================================
# Manual Nutrition Target Generator
# =====================================================

def generate_manual_targets(
    calories,
    protein,
    fat=None,
    carbs=None
):

    if fat is None and carbs is None:

        fat = round(
            (calories * 0.25) / 9,
            1
        )

        remaining_calories = (
            calories
            - (protein * 4)
            - (fat * 9)
        )

        carbs = round(
            remaining_calories / 4,
            1
        )

    elif fat is not None and carbs is None:

        remaining_calories = (
            calories
            - (protein * 4)
            - (fat * 9)
        )

        carbs = round(
            remaining_calories / 4,
            1
        )

    elif fat is None and carbs is not None:

        remaining_calories = (
            calories
            - (protein * 4)
            - (carbs * 4)
        )

        fat = round(
            remaining_calories / 9,
            1
        )

    return {
        "Calories": calories,
        "Protein_g": protein,
        "Fat_g": fat,
        "Carbs_g": carbs
    }


# =====================================================
# Macro Validation
# =====================================================

def validate_macros(
    calories,
    protein,
    fat,
    carbs,
    tolerance=50
):

    macro_calories = (
        protein * 4 +
        fat * 9 +
        carbs * 4
    )

    difference = round(
        abs(calories - macro_calories),
        1
    )

    return {
        "Target Calories": calories,
        "Macro Calories": round(macro_calories, 1),
        "Difference": difference,
        "Valid": difference <= tolerance
    }


# =====================================================
# Nutrition Quality Assessment
# =====================================================

def assess_macro_quality(
    calories,
    protein,
    fat,
    carbs
):

    warnings = []

    protein_pct = (protein * 4 / calories) * 100
    fat_pct = (fat * 9 / calories) * 100
    carb_pct = (carbs * 4 / calories) * 100

    if protein_pct < 15:
        warnings.append(
            "Protein intake appears lower than recommended for active individuals."
        )

    elif protein_pct > 40:
        warnings.append(
            "Protein intake appears unusually high."
        )

    if fat_pct < 20:
        warnings.append(
            "Fat intake appears lower than recommended."
        )

    elif fat_pct > 35:
        warnings.append(
            "Fat intake appears higher than recommended."
        )

    if carb_pct < 20:
        warnings.append(
            "Carbohydrate intake appears lower than recommended."
        )

    return {
        "Protein_%": round(protein_pct, 1),
        "Fat_%": round(fat_pct, 1),
        "Carbohydrates_%": round(carb_pct, 1),
        "Warnings": warnings if warnings else ["No Warnings"]
    }


# =====================================================
# Recommendation Eligibility
# =====================================================

def check_recommendation_eligibility(
    calories,
    protein,
    fat,
    carbs
):

    eligible = True

    reasons = []

    protein_pct = (
        protein * 4 / calories
    ) * 100

    fat_pct = (
        fat * 9 / calories
    ) * 100

    carb_pct = (
        carbs * 4 / calories
    ) * 100

    if protein_pct < 10:

        eligible = False

        reasons.append(
            "Protein target is extremely low."
        )

    if fat_pct < 10:

        eligible = False

        reasons.append(
            "Fat target is extremely low."
        )

    if carb_pct < 5:

        eligible = False

        reasons.append(
            "Carbohydrate target is extremely low."
        )

    return {
        "Eligible": eligible,
        "Reasons": reasons if reasons else ["No Issues"]
    }



__all__ = [
    "calculate_bmr",
    "calculate_tdee",
    "generate_calorie_targets",
    "calculate_protein",
    "calculate_fat",
    "calculate_carbs",
    "generate_macro_targets",
    "generate_all_goal_profiles",
    "generate_manual_targets",
    "validate_macros",
    "assess_macro_quality",
    "check_recommendation_eligibility"
]