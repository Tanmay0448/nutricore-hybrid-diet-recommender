"""Hybrid recommendation service used by the NutriCore Streamlit application."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


_REQUIRED_RECIPE_COLUMNS = {
    "RecipeId",
    "Name",
    "RecipeServings",
    "DietType",
    "CaloriesPerServing",
    "ProteinPerServing",
    "CarbsPerServing",
    "FatPerServing",
}


def _filter_by_diet(recipes_df: pd.DataFrame, diet_type: str) -> pd.DataFrame:
    if diet_type == "Vegetarian":
        return recipes_df[recipes_df["DietType"] == "Vegetarian"].copy()
    if diet_type == "Eggitarian":
        return recipes_df[recipes_df["DietType"].isin(["Vegetarian", "Eggitarian"])].copy()
    return recipes_df.copy()


def _meal_targets(calories: float, protein: float, carbs: float, fat: float, meals_remaining: int) -> dict[str, float]:
    return {
        "Calories": calories / meals_remaining,
        "Protein": protein / meals_remaining,
        "Carbs": carbs / meals_remaining,
        "Fat": fat / meals_remaining,
    }


def _nutrition_distance(values: dict[str, float], targets: dict[str, float]) -> float:
    weights = {"Calories": 0.30, "Protein": 0.40, "Carbs": 0.15, "Fat": 0.15}
    return sum(weights[macro] * abs(values[macro] - targets[macro]) / targets[macro] for macro in weights)


def _optimise_serving(recipe: pd.Series, targets: dict[str, float]) -> dict[str, float]:
    best = None
    servings = range(1, max(1, int(recipe["RecipeServings"])) + 1)
    for serving in servings:
        nutrients = {
            "Calories": recipe["CaloriesPerServing"] * serving,
            "Protein": recipe["ProteinPerServing"] * serving,
            "Carbs": recipe["CarbsPerServing"] * serving,
            "Fat": recipe["FatPerServing"] * serving,
        }
        distance = _nutrition_distance(nutrients, targets)
        if best is None or distance < best["Distance"]:
            best = {
                "RecipeId": recipe["RecipeId"],
                "Name": recipe["Name"],
                "BestServing": serving,
                "RecommendedCalories": round(nutrients["Calories"], 2),
                "RecommendedProtein": round(nutrients["Protein"], 2),
                "RecommendedCarbs": round(nutrients["Carbs"], 2),
                "RecommendedFat": round(nutrients["Fat"], 2),
                "Distance": distance,
                "NutritionScore": np.exp(-distance),
            }
    return best


def _nutrition_recommendations(
    recipes_df: pd.DataFrame,
    calories: float,
    protein: float,
    carbs: float,
    fat: float,
    meals_remaining: int,
    diet_type: str,
) -> pd.DataFrame:
    missing_columns = _REQUIRED_RECIPE_COLUMNS.difference(recipes_df.columns)
    if missing_columns:
        raise ValueError(f"Recipes are missing required columns: {sorted(missing_columns)}")
    targets = _meal_targets(calories, protein, carbs, fat, meals_remaining)
    candidates = _filter_by_diet(recipes_df, diet_type)
    recommendations = [_optimise_serving(recipe, targets) for _, recipe in candidates.iterrows()]
    return pd.DataFrame(recommendations).sort_values("Distance").reset_index(drop=True)


def _filter_collaborative_reviews(reviews_df: pd.DataFrame, minimum_ratings: int = 5) -> pd.DataFrame:
    active_users = reviews_df.groupby("AuthorId").size().loc[lambda counts: counts >= minimum_ratings].index
    active_recipes = reviews_df.groupby("RecipeId").size().loc[lambda counts: counts >= minimum_ratings].index
    return reviews_df[reviews_df["AuthorId"].isin(active_users) & reviews_df["RecipeId"].isin(active_recipes)].copy()


def _svd_scores(user_id, model, reviews_df: pd.DataFrame) -> pd.DataFrame:
    rated_recipe_ids = set(reviews_df.loc[reviews_df["AuthorId"] == user_id, "RecipeId"])
    candidates = set(reviews_df["RecipeId"]) - rated_recipe_ids
    predictions = [(recipe_id, model.predict(uid=user_id, iid=recipe_id, clip=False).est) for recipe_id in candidates]
    scores = pd.DataFrame(predictions, columns=["RecipeId", "PreferenceScore"])
    if scores.empty:
        return pd.DataFrame(columns=["RecipeId", "SVDScore"])
    scores["SVDScore"] = scores["PreferenceScore"].rank(pct=True, method="average")
    return scores[["RecipeId", "SVDScore"]]


def _similarity_scores(recipe_id, recipe_id_to_index: dict, tfidf_matrix) -> np.ndarray:
    if recipe_id not in recipe_id_to_index:
        raise ValueError(f"RecipeId '{recipe_id}' is not represented in the TF-IDF matrix.")
    recipe_index = recipe_id_to_index[recipe_id]
    return linear_kernel(tfidf_matrix[recipe_index], tfidf_matrix).flatten()


def _content_scores(liked_recipe_ids: list, recipes_df: pd.DataFrame, recipe_id_to_index: dict, tfidf_matrix) -> pd.DataFrame:
    if not liked_recipe_ids:
        raise ValueError("At least one liked recipe is required for content scoring.")
    available_ids = [recipe_id for recipe_id in liked_recipe_ids if recipe_id in recipe_id_to_index]
    if not available_ids:
        raise ValueError("No liked recipes are represented in the content model.")
    similarity_matrix = np.vstack([_similarity_scores(recipe_id, recipe_id_to_index, tfidf_matrix) for recipe_id in available_ids])
    for row_index, recipe_id in enumerate(available_ids):
        similarity_matrix[row_index, recipe_id_to_index[recipe_id]] = 0
    return pd.DataFrame({"RecipeId": recipes_df["RecipeId"], "ContentScore": similarity_matrix.max(axis=0)})


def _user_content_scores(
    user_id,
    recipes_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    recipe_id_to_index: dict,
    tfidf_matrix,
    minimum_rating: float = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive_reviews = reviews_df[(reviews_df["AuthorId"] == user_id) & (reviews_df["Rating"] >= minimum_rating)].copy()
    if positive_reviews.empty:
        raise ValueError("The user has no liked recipes in the historical review data.")
    liked_recipe_ids = positive_reviews["RecipeId"].drop_duplicates().tolist()
    scores = _content_scores(liked_recipe_ids, recipes_df, recipe_id_to_index, tfidf_matrix)
    liked_recipes = positive_reviews.merge(recipes_df[["RecipeId", "Name"]], on="RecipeId", how="left").sort_values("Rating", ascending=False)
    return liked_recipes[["RecipeId", "Name", "Rating"]], scores


def generate_hybrid_recommendations(
    recipes_df: pd.DataFrame,
    hybrid_static_df: pd.DataFrame,
    calories: float,
    protein: float,
    carbs: float,
    fat: float,
    meals_per_day: int,
    diet_type: str = "Non-Vegetarian",
    top_n: int | None = 10,
) -> pd.DataFrame:
    """Rank recipes by nutritional fit combined with static personalized scores."""
    nutrition_df = _nutrition_recommendations(recipes_df, calories, protein, carbs, fat, meals_per_day, diet_type)
    recommendations = nutrition_df.merge(hybrid_static_df, on="RecipeId", how="left")
    recommendations["StaticScore"] = recommendations["StaticScore"].fillna(0)
    recommendations["FinalScore"] = (0.45 * recommendations["NutritionScore"]) + recommendations["StaticScore"]
    recommendations = recommendations.sort_values("FinalScore", ascending=False).reset_index(drop=True)
    return recommendations if top_n is None else recommendations.head(top_n)


def build_hybrid_static_df(
    user_id,
    recipes_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    svd_model,
    tfidf_matrix,
    recipe_id_to_index: dict,
) -> tuple[pd.DataFrame, pd.DataFrame | None, bool]:
    """Build cached static hybrid scores for the active user and planning session."""
    popularity = pop_df[["RecipeId", "PopularityScore"]].copy()
    try:
        liked_recipes, content = _user_content_scores(user_id, recipes_df, reviews_df, recipe_id_to_index, tfidf_matrix)
        has_liked_recipes = True
    except ValueError:
        liked_recipes = None
        content = pd.DataFrame({"RecipeId": recipes_df["RecipeId"], "ContentScore": 0.0})
        has_liked_recipes = False

    collaborative_reviews = _filter_collaborative_reviews(reviews_df)
    svd = _svd_scores(user_id, svd_model, collaborative_reviews)
    static_df = popularity.merge(content, on="RecipeId", how="left").merge(svd, on="RecipeId", how="left")
    static_df["PopularityScore"] = static_df["PopularityScore"].fillna(static_df["PopularityScore"].mean())
    static_df["ContentScore"] = static_df["ContentScore"].fillna(0)
    static_df["SVDScore"] = static_df["SVDScore"].fillna(0)
    static_df["StaticScore"] = (
        (0.30 * static_df["PopularityScore"])
        + (0.20 * static_df["ContentScore"])
        + (0.05 * static_df["SVDScore"])
    )
    return static_df[["RecipeId", "StaticScore"]], liked_recipes, has_liked_recipes


__all__ = ["build_hybrid_static_df", "generate_hybrid_recommendations"]
