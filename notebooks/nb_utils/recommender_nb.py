import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import linear_kernel


# =====================================================
# Dataset Loading
# =====================================================

def load_dataset(filepath):
    return pd.read_parquet(filepath)


# =====================================================
# Diet Filtering
# =====================================================

def filter_by_diet(df, diet_type):

    if diet_type == "Vegetarian":

        return df[
            df["DietType"] == "Vegetarian"
        ].copy()

    elif diet_type == "Eggitarian":

        return df[
            df["DietType"].isin(
                ["Vegetarian", "Eggitarian"]
            )
        ].copy()

    return df.copy()


# =====================================================
# Meal Targets
# =====================================================

def calculate_meal_targets(
    calories,
    protein,
    carbs,
    fat,
    meals_per_day
):

    return {
        "Calories": calories / meals_per_day,
        "Protein": protein / meals_per_day,
        "Carbs": carbs / meals_per_day,
        "Fat": fat / meals_per_day
    }


# =====================================================
# Distance Score
# =====================================================

def calculate_distance(
    calories,
    protein,
    carbs,
    fat,
    target_calories,
    target_protein,
    target_carbs,
    target_fat,
    calorie_weight=0.30,
    protein_weight=0.40,
    carb_weight=0.15,
    fat_weight=0.15
):

    calorie_diff = (
        abs(calories - target_calories)
        / target_calories
    )

    protein_diff = (
        abs(protein - target_protein)
        / target_protein
    )

    carb_diff = (
        abs(carbs - target_carbs)
        / target_carbs
    )

    fat_diff = (
        abs(fat - target_fat)
        / target_fat
    )

    distance = (
        calorie_weight * calorie_diff
        + protein_weight * protein_diff
        + carb_weight * carb_diff
        + fat_weight * fat_diff
    )

    return distance


# =====================================================
# Serving Size Optimization
# =====================================================

def optimize_recipe(
    recipe_row,
    target_calories,
    target_protein,
    target_carbs,
    target_fat
):

    serving_sizes = np.arange(
        1,
        int(recipe_row["RecipeServings"]) + 1
    )

    best_distance = np.inf
    best_serving = None

    best_calories = None
    best_protein = None
    best_carbs = None
    best_fat = None

    for serving in serving_sizes:

        calories = (
            recipe_row["CaloriesPerServing"]
            * serving
        )

        protein = (
            recipe_row["ProteinPerServing"]
            * serving
        )

        carbs = (
            recipe_row["CarbsPerServing"]
            * serving
        )

        fat = (
            recipe_row["FatPerServing"]
            * serving
        )

        distance = calculate_distance(
            calories,
            protein,
            carbs,
            fat,
            target_calories,
            target_protein,
            target_carbs,
            target_fat
        )

        if distance < best_distance:

            best_distance = distance
            best_serving = serving

            best_calories = calories
            best_protein = protein
            best_carbs = carbs
            best_fat = fat

     
    return {
    "RecipeId": recipe_row["RecipeId"],
    "Name": recipe_row["Name"],
    "BestServing": best_serving,
    "RecommendedCalories": round(best_calories, 2),
    "RecommendedProtein": round(best_protein, 2),
    "RecommendedCarbs": round(best_carbs, 2),
    "RecommendedFat": round(best_fat, 2),
    "Distance": round(best_distance, 4),
    "NutritionScore": round(np.exp(-best_distance),4)
    }


# =====================================================
# Recommendation Engine
# =====================================================

def generate_recommendations(
    df,
    calories,
    protein,
    carbs,
    fat,
    meals_per_day,
    diet_type="Non-Vegetarian",
    top_n=None
):

    required_columns = [
        "RecipeId",
        "Name",
        "RecipeServings",
        "DietType",
        "CaloriesPerServing",
        "ProteinPerServing",
        "CarbsPerServing",
        "FatPerServing"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    candidate_df = filter_by_diet(
        df,
        diet_type
    )

    meal_targets = calculate_meal_targets(
        calories,
        protein,
        carbs,
        fat,
        meals_per_day
    )

    recommendations = []

    for _, row in candidate_df.iterrows():

        recommendations.append(
            optimize_recipe(
                recipe_row=row,
                target_calories=meal_targets["Calories"],
                target_protein=meal_targets["Protein"],
                target_carbs=meal_targets["Carbs"],
                target_fat=meal_targets["Fat"]
            )
        )
        
    recommendation_df = pd.DataFrame(recommendations)

    recommendation_df = (
    recommendation_df
    .sort_values("Distance")
    .reset_index(drop=True)
    )

    if top_n is not None:
        recommendation_df = recommendation_df.head(top_n)

    return recommendation_df


# =====================================================
# Meal Planner Initialization
# =====================================================

def initialize_meal_plan(
    calories,
    protein,
    carbs,
    fat,
    meals_per_day
):
    """
    Initializes the meal planner state.

    Parameters
    ----------
    calories, protein, carbs, fat : float
        Daily nutritional targets.

    meals_per_day : int
        Total number of meals to plan.

    Returns
    -------
    dict
        Initial meal planner state.
    """

    planner = {
        "remaining_calories": calories,
        "remaining_protein": protein,
        "remaining_carbs": carbs,
        "remaining_fat": fat,
        "remaining_meals": meals_per_day,
        "selected_recipe_ids": [],
        "meals": []
    }

    return planner


# =====================================================
# Update Meal Planner
# =====================================================

def update_meal_plan(
    planner,
    selected_recipe,
    meal_number
):
    """
    Updates the meal planner after a recipe has been selected.

    Parameters
    ----------
    planner : dict
        Current meal planner state.

    selected_recipe : pd.Series
        Selected recipe.

    meal_number : int
        Current meal number (zero-based).

    Returns
    -------
    dict
        Updated planner state.
    """

    selected_recipe = selected_recipe.copy()

    selected_recipe["Meal"] = meal_number + 1

    planner["meals"].append(
        selected_recipe.to_dict()
    )

    planner["selected_recipe_ids"].append(
        selected_recipe["RecipeId"]
    )

    planner["remaining_calories"] -= (
        selected_recipe["RecommendedCalories"]
    )

    planner["remaining_protein"] -= (
        selected_recipe["RecommendedProtein"]
    )

    planner["remaining_carbs"] -= (
        selected_recipe["RecommendedCarbs"]
    )

    planner["remaining_fat"] -= (
        selected_recipe["RecommendedFat"]
    )

    planner["remaining_meals"] -= 1

    return planner


# =====================================================
# Daily Meal Planner
# =====================================================

def plan_daily_meals(
    recipes_df,
    calories,
    protein,
    carbs,
    fat,
    meals_per_day,
    diet_type="Non-Vegetarian",
    selection_indices=None,
    top_n=None
):
    """
    Generates a complete daily meal plan using the recipe
    recommendation engine.

    Parameters
    ----------
    recipes_df : pd.DataFrame
        Processed recipe dataset.

    calories, protein, carbs, fat : float
        Daily nutritional targets.

    meals_per_day : int
        Number of meals to generate.

    diet_type : str, default="Non-Vegetarian"
        Dietary preference.

    selection_indices : list, optional
        Index of the recommendation selected for each meal.

        Example
        -------
        [0, 0, 2, 1]

        0 -> Best recommendation
        2 -> Third recommendation

        If None, the highest-ranked recommendation is
        automatically selected for every meal.

    top_n : int
        Number of recommendations generated for each meal.

    Returns
    -------
    dict
        Meal planner containing:

        - Planned meals
        - Remaining nutritional targets
        - Selected recipe IDs
    """

    planner = initialize_meal_plan(
        calories,
        protein,
        carbs,
        fat,
        meals_per_day
    )

    # =====================================================
    # Validate User Selections
    # =====================================================

    if selection_indices is None:

        selection_indices = [0] * meals_per_day

    elif len(selection_indices) < meals_per_day:

        selection_indices.extend(
            [0] * (meals_per_day - len(selection_indices))
        )

    elif len(selection_indices) > meals_per_day:

        raise ValueError(
            "The number of selection indices cannot exceed the number of meals."
        )

    # =====================================================
    # Meal Planning Loop
    # =====================================================

    for meal_number in range(meals_per_day):

        available_recipes = recipes_df[
            ~recipes_df["RecipeId"].isin(
                planner["selected_recipe_ids"]
            )
        ]

        recommendations = generate_recommendations(
            df=available_recipes,
            calories=planner["remaining_calories"],
            protein=planner["remaining_protein"],
            carbs=planner["remaining_carbs"],
            fat=planner["remaining_fat"],
            meals_per_day=planner["remaining_meals"],
            diet_type=diet_type,
            top_n=top_n
        )

        selected_index = selection_indices[meal_number]

        if selected_index >= len(recommendations):

            raise ValueError(
                f"Meal {meal_number + 1}: "
                f"Selection index {selected_index} is out of range. "
                f"Only {len(recommendations)} recommendation(s) are available."
            )

        selected_recipe = recommendations.iloc[
            selected_index
        ].copy()

        planner = update_meal_plan(
            planner,
            selected_recipe,
            meal_number
        )

    # =====================================================
    # Convert Meal Plan to DataFrame
    # =====================================================

    planner["meals"] = pd.DataFrame(
        planner["meals"]
    )

    return planner



# =====================================================
# Collaborative Filtering Dataset Preparation
# =====================================================

def filter_collaborative_data(
    reviews_df,
    min_user_ratings=5,
    min_recipe_ratings=5
):
    """
    Filter the interaction dataset for collaborative filtering.

    Users and recipes with fewer interactions than the specified
    thresholds are removed to reduce sparsity while preserving
    the original interaction statistics.

    Parameters
    ----------
    reviews_df : pandas.DataFrame
        User-recipe interaction dataset.

    min_user_ratings : int, default=5
        Minimum number of ratings required for a user.

    min_recipe_ratings : int, default=5
        Minimum number of ratings required for a recipe.

    Returns
    -------
    pandas.DataFrame
        Filtered interaction dataset.
    """

    # User activity (computed on original dataset)
    user_activity = (
        reviews_df
        .groupby("AuthorId")
        .size()
    )

    # Recipe activity (computed on original dataset)
    recipe_activity = (
        reviews_df
        .groupby("RecipeId")
        .size()
    )

    # Active users and recipes
    active_users = user_activity[
        user_activity >= min_user_ratings
    ].index

    active_recipes = recipe_activity[
        recipe_activity >= min_recipe_ratings
    ].index

    # Apply both filters simultaneously
    filtered_df = reviews_df[
        (reviews_df["AuthorId"].isin(active_users)) &
        (reviews_df["RecipeId"].isin(active_recipes))
    ].copy()

    return filtered_df

# =====================================================
# Singular Value Decomposition (SVD) Recommendation
# =====================================================

# =====================================================
# Singular Value Decomposition (SVD) Recommendation
# =====================================================

def recommend_svd(
    user_id,
    model,
    reviews_df,
    recipes_df,
    top_n=None
):
    """
    Generate personalized recipe recommendations using a
    trained Singular Value Decomposition (SVD) model.

    Parameters
    ----------
    user_id : int
        Target user ID.

    model : surprise.SVD
        Trained Surprise SVD model.

    reviews_df : pandas.DataFrame
        Collaborative filtering interaction dataset used
        to train the SVD model.

    recipes_df : pandas.DataFrame
        Recipe dataset containing recipe information.

    top_n : int, default=10
        Number of recommendations to return.

    Returns
    -------
    pandas.DataFrame
        Top-N collaborative recommendations containing
        the preference score and normalized SVD score.
    """

    # ---------------------------------------------
    # Recipes already rated by the user
    # ---------------------------------------------

    rated_recipes = set(
        reviews_df.loc[
            reviews_df["AuthorId"] == user_id,
            "RecipeId"
        ]
    )

    # ---------------------------------------------
    # Candidate recipes
    # ---------------------------------------------
    # Recommend only recipes present in the
    # collaborative filtering dataset while excluding
    # recipes already rated by the user.

    candidate_recipes = (
        set(reviews_df["RecipeId"])
        - rated_recipes
    )

    # ---------------------------------------------
    # Predict preference scores
    # ---------------------------------------------

    predictions = []

    for recipe_id in candidate_recipes:

        prediction = model.predict(
            uid=user_id,
            iid=recipe_id,
            clip=False
        )

        predictions.append(
            [
                recipe_id,
                prediction.est
            ]
        )

    recommendations = pd.DataFrame(
        predictions,
        columns=[
            "RecipeId",
            "PreferenceScore"
        ]
    )

    # ---------------------------------------------
    # Percentile-based collaborative score
    # ---------------------------------------------

    recommendations["SVDScore"] = (
        recommendations["PreferenceScore"]
        .rank(
            pct=True,
            method="average"
        )
    )

    # ---------------------------------------------
    # Select Top Recommendations
    # ---------------------------------------------

    recommendations = (
    recommendations
    .sort_values(
        by="SVDScore",
        ascending=False
    )
    .reset_index(drop=True)
    )

    if top_n is not None:

        recommendations = recommendations.head(top_n)

    # ---------------------------------------------
    # Add recipe names
    # ---------------------------------------------

    recommendations = recommendations.merge(
        recipes_df[
            [
                "RecipeId",
                "Name"
            ]
        ],
        on="RecipeId",
        how="left"
    )

    # ---------------------------------------------
    # Final Output
    # ---------------------------------------------

    recommendations = recommendations[
        [
            "RecipeId",
            "Name",
            "PreferenceScore",
            "SVDScore"
        ]
    ]

    return recommendations



# =====================================================
# Content-Based Similarity Computation
# =====================================================

def compute_similarity_scores(
    recipe_id,
    recipe_id_to_index,
    tfidf_matrix
):
    """
    Compute semantic similarity scores for a given recipe.

    Parameters
    ----------
    recipe_id : int or float
        Unique Recipe ID.

    recipe_id_to_index : pandas.Series
        Mapping from RecipeId to the corresponding row
        index in the TF-IDF matrix.

    tfidf_matrix : scipy.sparse.csr_matrix
        TF-IDF feature matrix representing recipes.

    Returns
    -------
    numpy.ndarray
        Similarity scores between the selected recipe
        and every recipe in the dataset.
    """

    if recipe_id not in recipe_id_to_index:

        raise ValueError(
            f"RecipeId '{recipe_id}' not found."
        )

    recipe_index = recipe_id_to_index[recipe_id]

    similarity_scores = linear_kernel(
        tfidf_matrix[recipe_index],
        tfidf_matrix
    ).flatten()

    return similarity_scores



# =====================================================
# User Content Profile
# =====================================================

def compute_content_scores(
    liked_recipe_ids,
    recipes_df,
    recipe_id_to_index,
    tfidf_matrix,
    return_details=False
):
    """
    Compute aggregated content similarity scores for a user.

    Parameters
    ----------
    liked_recipe_ids : list
        List of positively rated RecipeIds.

    recipes_df : pandas.DataFrame
        Recipe dataset.

    recipe_id_to_index : pandas.Series
        Mapping from RecipeId to the corresponding row
        index in the TF-IDF matrix.

    tfidf_matrix : scipy.sparse.csr_matrix
        TF-IDF feature matrix representing recipes.

    return_details : bool, default=False
        If True, return recipe information along with
        the computed content scores.

    Returns
    -------
    pandas.DataFrame

        If return_details=False

            RecipeId
            ContentScore

        If return_details=True

            RecipeId
            Name
            RecipeCategory
            DietType
            ContentScore
    """

    if len(liked_recipe_ids) == 0:

        raise ValueError(
            "At least one liked recipe is required."
        )

    similarity_matrix = []

    for recipe_id in liked_recipe_ids:

        similarity_matrix.append(
            compute_similarity_scores(
                recipe_id=recipe_id,
                recipe_id_to_index=recipe_id_to_index,
                tfidf_matrix=tfidf_matrix
            )
        )

    similarity_matrix = np.vstack(
        similarity_matrix
    )
    
    # 🔥 Remove self similarity
    for i, recipe_id in enumerate(liked_recipe_ids):
        idx = recipe_id_to_index[recipe_id]
        similarity_matrix[i, idx] = 0

    content_scores = similarity_matrix.max(
        axis=0
    )

    content_df = pd.DataFrame({
        "RecipeId": recipes_df["RecipeId"],
        "ContentScore": content_scores
    })

    content_df = (
        content_df
        .sort_values(
            by="ContentScore",
            ascending=False
        )
        .reset_index(drop=True)
    )

    if not return_details:

        return content_df

    content_df = content_df.merge(
        recipes_df[
            [
                "RecipeId",
                "Name",
                "RecipeCategory",
                "DietType"
            ]
        ],
        on="RecipeId",
        how="left"
    )

    return content_df[
        [
            "RecipeId",
            "Name",
            "RecipeCategory",
            "DietType",
            "ContentScore"
        ]
    ]

# =====================================================
# Similar Recipe Recommendation
# =====================================================

def recommend_similar_recipes(
    recipe_name,
    recipes_df,
    recipe_id_to_index,
    tfidf_matrix,
    top_n=10
):
    """
    Retrieve the most semantically similar recipes.

    Parameters
    ----------
    recipe_name : str
        Name of the input recipe.

    recipes_df : pandas.DataFrame
        Recipe dataset.

    recipe_id_to_index : pandas.Series
        Mapping from RecipeId to the corresponding row
        index in the TF-IDF matrix.

    tfidf_matrix : scipy.sparse.csr_matrix
        TF-IDF feature matrix representing recipes.

    top_n : int, default=10
        Number of similar recipes to return.

    Returns
    -------
    pandas.DataFrame
        Top-N most similar recipes.
    """

    recipe = recipes_df[
        recipes_df["Name"] == recipe_name
    ]

    if recipe.empty:

        raise ValueError(
            f"Recipe '{recipe_name}' not found."
        )

    recipe_id = recipe.iloc[0]["RecipeId"]

    similarity_scores = compute_similarity_scores(
        recipe_id=recipe_id,
        recipe_id_to_index=recipe_id_to_index,
        tfidf_matrix=tfidf_matrix
    )

    recommendations = pd.DataFrame({
        "RecipeId": recipes_df["RecipeId"],
        "Name": recipes_df["Name"],
        "RecipeCategory": recipes_df["RecipeCategory"],
        "DietType": recipes_df["DietType"],
        "SimilarityScore": similarity_scores
    })

    recommendations = recommendations[
        recommendations["RecipeId"] != recipe_id
    ]

    recommendations = (
        recommendations
        .sort_values(
            by="SimilarityScore",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return recommendations


# =====================================================
# Content-Based Recommendations for User
# =====================================================

def compute_content_recommendations(
    user_id,
    recipes_df,
    reviews_df,
    recipe_id_to_index,
    tfidf_matrix,
    min_rating=4,
    top_n=None,
    return_liked_recipes=False,
    return_details=False
):
    """
    Generate content-based recommendation scores for a user.

    This function can return:
    - Full score table (for hybrid systems) when top_n=None
    - Top-N recommendations when top_n is specified

    Parameters
    ----------
    user_id : int
        Target user ID.

    recipes_df : pandas.DataFrame
        Recipe dataset.

    reviews_df : pandas.DataFrame
        User review dataset.

    recipe_id_to_index : dict or pandas.Series
        Mapping from RecipeId to TF-IDF matrix index.

    tfidf_matrix : scipy.sparse.csr_matrix
        TF-IDF feature matrix.

    min_rating : int, default=4
        Minimum rating to consider a recipe as liked.

    top_n : int or None, default=None
        Number of recommendations to return.
        If None, returns scores for all recipes.

    return_liked_recipes : bool, default=False
        If True, also returns liked recipes used for profiling.

    Returns
    -------
    pandas.DataFrame OR tuple

        If return_liked_recipes=False:
            DataFrame with recommendation scores.

        If return_liked_recipes=True:
            (liked_recipes_df, recommendations_df)
    """

    positive_reviews = reviews_df[
        (reviews_df["AuthorId"] == user_id) &
        (reviews_df["Rating"] >= min_rating)
    ].copy()

    if positive_reviews.empty:
        raise ValueError(
            f"User {user_id} has no recipes rated "
            f"{min_rating} or higher."
        )

    liked_recipe_ids = (
        positive_reviews["RecipeId"]
        .unique()
        .tolist()
    )

    recommendations = compute_content_scores(
        liked_recipe_ids=liked_recipe_ids,
        recipes_df=recipes_df,
        recipe_id_to_index=recipe_id_to_index,
        tfidf_matrix=tfidf_matrix,
        return_details=return_details
    )


    if top_n is not None:
        recommendations = recommendations.head(top_n)

    #recommendations = content_df.reset_index(drop=True)

    if not return_liked_recipes:
        return recommendations

    liked_recipes = (
        positive_reviews
        .merge(
            recipes_df[
                [
                    "RecipeId",
                    "Name",
                    "RecipeCategory",
                    "DietType"
                ]
            ],
            on="RecipeId",
            how="left"
        )
        .sort_values(by="Rating", ascending=False)
        [
            [
                "RecipeId",
                "Name",
                "RecipeCategory",
                "DietType",
                "Rating"
            ]
        ]
        .reset_index(drop=True)
    )

    return liked_recipes, recommendations


# =====================================================
# Hybrid Recommendation
# =====================================================

def generate_hybrid_recommendations(
    recipes_df,
    hybrid_static_df,
    calories,
    protein,
    carbs,
    fat,
    meals_per_day,
    diet_type="Non-Vegetarian",
    top_n=10
):
    """
    Generate hybrid recipe recommendations by combining
    nutritional relevance with precomputed static scores.

    StaticScore contains the weighted combination of:
    - Content-based similarity
    - Popularity
    - SVD

    The final hybrid score is:

        FinalScore =
            0.45 * NutritionScore +
            StaticScore

    Parameters
    ----------
    recipes_df : pd.DataFrame
        Recipe dataset.

    hybrid_static_df : pd.DataFrame
        Precomputed static recommendation scores containing
        RecipeId and StaticScore.

    calories, protein, carbs, fat : float
        Current nutritional targets.

    meals_per_day : int
        Number of meals remaining.

    diet_type : str, default="Non-Vegetarian"
        Dietary preference.

    top_n : int, default=10
        Number of recommendations to return.

    Returns
    -------
    pd.DataFrame
        Ranked hybrid recommendations.
    """

    nutrition_df = generate_recommendations(
        df=recipes_df,
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat,
        meals_per_day=meals_per_day,
        diet_type=diet_type
    )

    recommendation_df = nutrition_df.merge(
        hybrid_static_df[
            ["RecipeId", "StaticScore"]
        ],
        on="RecipeId",
        how="left"
    )

    recommendation_df["StaticScore"] = (
        recommendation_df["StaticScore"].fillna(0)
    )

    recommendation_df["FinalScore"] = (
        0.45 * recommendation_df["NutritionScore"]
        + recommendation_df["StaticScore"]
    )

    recommendation_df = (
        recommendation_df
        .sort_values(
            by="FinalScore",
            ascending=False
        )
        .reset_index(drop=True)
    )

    if top_n is not None:
        recommendation_df = recommendation_df.head(top_n)

    return recommendation_df


# =====================================================
# Automatic Hybrid Meal Planner
# =====================================================

def automatic_hybrid_meal_planner(
    recipes_df,
    hybrid_static_df,
    calories,
    protein,
    carbs,
    fat,
    meals_per_day,
    diet_type="Non-Vegetarian",
    top_n=10
):
    """
    Generate a complete daily meal plan automatically
    using the highest-ranked hybrid recommendation for
    each meal.

    Previously selected recipes are excluded from subsequent
    recommendations, and nutritional targets are updated after
    every meal.

    Parameters
    ----------
    recipes_df : pd.DataFrame
        Recipe dataset.

    hybrid_static_df : pd.DataFrame
        Precomputed hybrid static scores.

    calories, protein, carbs, fat : float
        Daily nutritional targets.

    meals_per_day : int
        Number of meals to generate.

    diet_type : str, default="Non-Vegetarian"
        Dietary preference.

    top_n : int, default=10
        Number of hybrid recommendations considered per meal.

    Returns
    -------
    dict
        Completed meal planner state.
    """

    planner = initialize_meal_plan(
        calories,
        protein,
        carbs,
        fat,
        meals_per_day
    )

    for meal_number in range(meals_per_day):

        available_recipes = recipes_df[
            ~recipes_df["RecipeId"].isin(
                planner["selected_recipe_ids"]
            )
        ]

        recommendations = generate_hybrid_recommendations(
            recipes_df=available_recipes,
            hybrid_static_df=hybrid_static_df,
            calories=planner["remaining_calories"],
            protein=planner["remaining_protein"],
            carbs=planner["remaining_carbs"],
            fat=planner["remaining_fat"],
            meals_per_day=planner["remaining_meals"],
            diet_type=diet_type,
            top_n=top_n
        )

        if recommendations.empty:
            raise ValueError(
                f"No hybrid recommendations available "
                f"for meal {meal_number + 1}."
            )

        selected_recipe = recommendations.iloc[0].copy()

        planner = update_meal_plan(
            planner,
            selected_recipe,
            meal_number
        )

    planner["meals"] = pd.DataFrame(
        planner["meals"]
    )

    return planner


# =====================================================
# Interactive Hybrid Meal Planner
# =====================================================

def interactive_hybrid_meal_planner(
    recipes_df,
    hybrid_static_df,
    calories,
    protein,
    carbs,
    fat,
    meals_per_day,
    diet_type="Non-Vegetarian",
    selection_indices=None,
    top_n=10
):
    """
    Generate a daily meal plan using user-selected hybrid
    recommendations.

    Parameters
    ----------
    recipes_df : pd.DataFrame
        Recipe dataset.

    hybrid_static_df : pd.DataFrame
        Precomputed hybrid static scores.

    calories, protein, carbs, fat : float
        Daily nutritional targets.

    meals_per_day : int
        Number of meals to generate.

    diet_type : str, default="Non-Vegetarian"
        Dietary preference.

    selection_indices : list of int
        Zero-based index of the selected recommendation
        for each meal.

    top_n : int, default=10
        Number of recommendations generated per meal.

    Returns
    -------
    dict
        Completed meal planner state.
    """

    if selection_indices is None:
        raise ValueError(
            "selection_indices must be provided for "
            "interactive meal planning."
        )

    if len(selection_indices) != meals_per_day:
        raise ValueError(
            "The number of selection indices must match "
            "the number of meals."
        )

    planner = initialize_meal_plan(
        calories,
        protein,
        carbs,
        fat,
        meals_per_day
    )

    for meal_number in range(meals_per_day):

        # Exclude previously selected recipes
        available_recipes = recipes_df[
            ~recipes_df["RecipeId"].isin(
                planner["selected_recipe_ids"]
            )
        ]

        # Generate hybrid recommendations
        recommendations = generate_hybrid_recommendations(
            recipes_df=available_recipes,
            hybrid_static_df=hybrid_static_df,
            calories=planner["remaining_calories"],
            protein=planner["remaining_protein"],
            carbs=planner["remaining_carbs"],
            fat=planner["remaining_fat"],
            meals_per_day=planner["remaining_meals"],
            diet_type=diet_type,
            top_n=top_n
        )

        if recommendations.empty:
            raise ValueError(
                f"No hybrid recommendations available "
                f"for meal {meal_number + 1}."
            )

        # User-selected recommendation
        selected_index = selection_indices[
            meal_number
        ]

        if selected_index < 0 or selected_index >= len(
            recommendations
        ):
            raise ValueError(
                f"Meal {meal_number + 1}: "
                f"selection index {selected_index} is out "
                f"of range. Available indices: "
                f"0 to {len(recommendations) - 1}."
            )

        selected_recipe = recommendations.iloc[
            selected_index
        ].copy()

        # Update planner state
        planner = update_meal_plan(
            planner,
            selected_recipe,
            meal_number
        )

    planner["meals"] = pd.DataFrame(
        planner["meals"]
    )

    return planner



# =====================================================
# Export Functions
# =====================================================

__all__ = [
    "load_dataset",
    "filter_by_diet",
    "calculate_meal_targets",
    "calculate_distance",
    "optimize_recipe",
    "generate_recommendations",
    "plan_daily_meals",

    "filter_collaborative_data",
    "recommend_svd",

    "compute_similarity_scores",
    "compute_content_scores",
    "recommend_similar_recipes",
    "compute_content_recommendations",
    
    "generate_hybrid_recommendations",
    "automatic_hybrid_meal_planner",
    "interactive_hybrid_meal_planner"
]