import os
import hashlib
import hmac
import secrets
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# -------------------------
# DATABASE URL
# -------------------------
def _database_url():
    try:
        import streamlit as st
        if "DATABASE_URL" in st.secrets:
            return st.secrets["DATABASE_URL"]
    except Exception:
        pass

    return os.getenv("DATABASE_URL", "sqlite:///nutricore.db")


def get_engine() -> Engine:
    return create_engine(_database_url(), pool_pre_ping=True, future=True)


# -------------------------
# PASSWORD
# -------------------------
def hash_passcode(passcode: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac("sha256", passcode.encode(), salt, 200_000)
    return f"{salt.hex()}${key.hex()}"


def _verify_passcode(passcode: str, stored_hash: str) -> bool:
    try:
        salt_hex, key_hex = stored_hash.split("$")
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(key_hex)
    except:
        return False

    actual = hashlib.pbkdf2_hmac("sha256", passcode.encode(), salt, 200_000)
    return hmac.compare_digest(actual, expected)


# -------------------------
# INIT DB
# -------------------------
def initialize_database(engine: Engine):

    with engine.begin() as conn:

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            passcode_hash TEXT,
            user_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id INTEGER,
            user_id TEXT,
            rating REAL,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, recipe_id)
        )
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS blocked_recipes (
            user_id TEXT,
            recipe_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, recipe_id)
        )
        """))


# -------------------------
# USER FUNCTIONS
# -------------------------
def _user_exists(engine, user_id):
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT 1 FROM users WHERE user_id=:u"),
            {"u": user_id}
        ).first() is not None


def create_user(engine, user_id, passcode):
    if _user_exists(engine, user_id):
        return False

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO users (user_id, passcode_hash, user_type)
            VALUES (:u, :p, 'app')
        """), {
            "u": user_id,
            "p": hash_passcode(passcode)
        })
    return True


def authenticate_user(engine, user_id, passcode):
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT passcode_hash FROM users WHERE user_id=:u"),
            {"u": user_id}
        ).first()

    return bool(row and _verify_passcode(passcode, row[0]))


# -------------------------
# RATINGS
# -------------------------
def upsert_rating(engine, user_id, recipe_id, rating):
    with engine.begin() as conn:
        conn.execute(text("""
        INSERT INTO reviews (recipe_id, user_id, rating, source)
        VALUES (:r, :u, :rating, 'app')
        ON CONFLICT(user_id, recipe_id)
        DO UPDATE SET rating = excluded.rating
        """), {
            "r": recipe_id,
            "u": user_id,
            "rating": rating
        })


def delete_rating(engine, user_id, recipe_id):
    with engine.begin() as conn:
        conn.execute(text("""
        DELETE FROM reviews
        WHERE user_id=:u AND recipe_id=:r
        """), {"u": user_id, "r": recipe_id})


def get_user_ratings(engine, user_id):
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT recipe_id, rating
            FROM reviews
            WHERE user_id=:u
        """), {"u": user_id}).fetchall()

    return {r[0]: r[1] for r in rows}


# -------------------------
# BLOCK SYSTEM
# -------------------------
def block_recipe(engine, user_id, recipe_id):

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT OR IGNORE INTO blocked_recipes (user_id, recipe_id)
            VALUES (:u, :r)
        """), {
            "u": user_id,
            "r": recipe_id
        })


def get_blocked_ids(engine, user_id):

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT recipe_id FROM blocked_recipes
            WHERE user_id=:u
        """), {"u": user_id}).fetchall()

    return {r[0] for r in rows}


def unblock_recipe(engine, user_id, recipe_id):
    """Remove a recipe from a user's saved block list."""
    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM blocked_recipes
            WHERE user_id=:u AND recipe_id=:r
        """), {"u": user_id, "r": recipe_id})


__all__ = [
    "authenticate_user",
    "block_recipe",
    "create_user",
    "delete_rating",
    "get_blocked_ids",
    "get_engine",
    "get_user_ratings",
    "hash_passcode",
    "initialize_database",
    "unblock_recipe",
    "upsert_rating",
]
