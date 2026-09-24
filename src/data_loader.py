"""Cached loading of NutriCore datasets and trained recommendation artifacts."""

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz


@st.cache_resource(show_spinner=False)
def load_recommendation_assets():
    """Load the datasets, model artifacts, and recipe-index mapping used by the app."""
    recipes_df = pd.read_parquet("data/processed/recipes_processed.parquet")
    reviews_df = pd.read_parquet("data/processed/reviews_processed.parquet")
    popularity_df = pd.read_parquet("data/processed/recipe_popularity.parquet")
    svd_model = joblib.load("models/svd_model.pkl")
    tfidf_matrix = load_npz("models/tfidf_matrix.npz")
    recipe_id_to_index = {recipe_id: index for index, recipe_id in enumerate(recipes_df["RecipeId"])}
    return recipes_df, reviews_df, popularity_df, svd_model, tfidf_matrix, recipe_id_to_index


__all__ = ["load_recommendation_assets"]
