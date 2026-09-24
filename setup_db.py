import pandas as pd
from src.database import get_engine, hash_passcode, initialize_database

print("Starting DB setup...")

# -------- LOAD DATA --------
print("Loading parquet...")
reviews_df = pd.read_parquet("data/processed/reviews_processed.parquet")
print(f"Loaded data: {len(reviews_df)} rows")

required_cols = ["AuthorId", "RecipeId", "Rating"]
for col in required_cols:
    if col not in reviews_df.columns:
        raise ValueError(f"Missing column: {col}")

print("Columns check passed")

# -------- ENGINE --------
engine = get_engine()
initialize_database(engine)
print("Database initialized")

# -------- USERS --------
print("Processing users...")

users_df = (
    reviews_df[["AuthorId"]]
    .dropna()
    .astype(str)
    .drop_duplicates()
)

print(f"Unique users: {len(users_df)}")

users_df.columns = ["user_id"]

print("Generating default hash once...")
default_hash = hash_passcode("0000")

users_df["passcode_hash"] = default_hash
users_df["user_type"] = "foodcom"

print("Inserting users...")
users_df.to_sql(
    "users",
    engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=5000
)

print("Users inserted")

# -------- REVIEWS --------
print("Processing reviews...")

reviews_clean = reviews_df[["RecipeId", "AuthorId", "Rating"]].dropna()
print(f"Valid reviews: {len(reviews_clean)}")

reviews_clean = reviews_clean.astype({
    "RecipeId": int,
    "AuthorId": str,
    "Rating": float
})

reviews_clean.columns = ["recipe_id", "user_id", "rating"]
reviews_clean["source"] = "dataset"

print("Inserting reviews...")

reviews_clean.to_sql(
    "reviews",
    engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=5000
)

print("Reviews inserted")

print("DB setup complete")
