"""Small, framework-independent helpers for the NutriCore app."""

import re


MACROS = ("Calories", "Protein", "Carbs", "Fat")


def display_number(value: float) -> int | float:
    """Return whole numeric values without an unnecessary decimal suffix."""
    numeric_value = float(value)
    return int(numeric_value) if numeric_value.is_integer() else numeric_value


def generate_foodcom_url(name: str, recipe_id: int | float, serving: int | float | None = None) -> str:
    """Build a Food.com recipe URL using a punctuation-free URL slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    url = f"https://www.food.com/recipe/{slug}-{display_number(recipe_id)}"
    return f"{url}?scale={display_number(serving)}" if serving is not None else url


def parse_optional_whole_number(value: int | None) -> int | None:
    """Preserve an empty numeric field while normalising supplied values to integers."""
    return None if value is None else int(value)


def recipe_name(recipe_id: int | float, recipe_names: dict) -> str:
    """Resolve a recipe ID to a display name with a useful fallback."""
    return recipe_names.get(recipe_id, f"Recipe {display_number(recipe_id)}")


def preference_recipe_url(recipe_id: int | float, recipe_names: dict) -> str:
    """Create a Food.com link for a saved recipe without a serving-size query."""
    return generate_foodcom_url(recipe_name(recipe_id, recipe_names), display_number(recipe_id))


def plan_totals(plan) -> dict[str, float]:
    """Sum the recommended nutrition columns in a selected meal plan."""
    return {macro: plan[f"Recommended{macro}"].sum() for macro in MACROS}


__all__ = [
    "MACROS",
    "display_number",
    "generate_foodcom_url",
    "parse_optional_whole_number",
    "plan_totals",
    "preference_recipe_url",
    "recipe_name",
]
