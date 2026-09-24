import html

import pandas as pd
import streamlit as st

import src.recommender as recommender
from src.data_loader import load_recommendation_assets
from src.database import (
    authenticate_user, block_recipe, create_user, delete_rating, get_blocked_ids,
    get_engine, get_user_ratings, initialize_database, unblock_recipe, upsert_rating,
)
from src.nutrition import (
    assess_macro_quality, calculate_bmr, calculate_tdee, check_recommendation_eligibility,
    generate_all_goal_profiles, generate_calorie_targets, generate_manual_targets, validate_macros,
)
from src.ui import (
    render_blocked_recipe_editor, render_brand, render_final_meal_hint, render_loading_overlay,
    render_macro_columns, render_meal_cards, render_plan_meals, render_plan_summary,
    render_preference_navigation, render_rated_recipe_editor, restore_selected_recipe,
)
from src.utils import parse_optional_whole_number, plan_totals


st.set_page_config(page_title="NutriCore", page_icon="🥗", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
.block-container {width: 100%; max-width: 1120px; box-sizing: border-box; margin-left: auto; margin-right: auto; padding-top: 2.25rem; padding-bottom: 2.25rem;}
html, body, [class*="css"] {font-family: "DM Sans", "Segoe UI", sans-serif;}
.hero {text-align: center; padding: 1.5rem 0 2rem;}
.hero h1 {font-family: "Playfair Display", Georgia, serif; font-size: clamp(2.7rem, 6vw, 4.2rem); letter-spacing: -.045em; margin-bottom: .15rem;}
.hero p {font-family: "DM Sans", "Segoe UI", sans-serif; font-size: 1.05rem; letter-spacing: .08em; text-transform: uppercase; color: #64748b;}
.recipe-panel {padding: 1.25rem 1.4rem; border: 1px solid rgba(128,128,128,.24); border-radius: 16px; margin-top: .8rem;}
.macro-label {color: #6b7280; font-size: .85rem; margin-bottom: 0;}
.macro-value {font-size: 1.25rem; font-weight: 650; margin-top: -.35rem;}
.loader-overlay {position: fixed; inset: 0; z-index: 9999; display: flex; align-items: center; justify-content: center; background: rgba(250, 251, 252, .97); pointer-events: auto; touch-action: none; cursor: progress;}
.loader-card {width: min(405px, calc(100vw - 48px)); text-align: center; padding: 2rem 2rem 2.2rem; border-radius: 28px; background: linear-gradient(145deg, #ffffff, #f4fbf6); box-shadow: 0 20px 54px rgba(31, 41, 55, .15);}
.loader-scene {position: relative; width: 190px; height: 118px; margin: 0 auto .75rem; overflow: hidden;}
.loader-plate {position: absolute; left: 50%; bottom: 2px; width: 92px; height: 36px; transform: translateX(-50%); border-radius: 50%; background: #d7eee0; box-shadow: inset 0 -8px 0 #afd6bc, 0 7px 13px rgba(30, 91, 52, .16);}
.loader-salad {position: absolute; left: 50%; bottom: 14px; width: 68px; height: 50px; transform: translateX(-50%); border-radius: 50% 50% 42% 42%; background: radial-gradient(circle at 30% 35%, #8fd17b 0 9%, transparent 10%), radial-gradient(circle at 70% 30%, #f6c85f 0 9%, transparent 10%), radial-gradient(circle at 52% 58%, #e86e5a 0 10%, transparent 11%), #54a76d; box-shadow: inset 0 -7px 0 #3b8557; animation: salad-bounce 1.7s ease-in-out infinite;}
.loader-food {position: absolute; font-size: 1.55rem; filter: drop-shadow(0 4px 3px rgba(31, 41, 55, .12)); animation: ingredient-float 2.1s ease-in-out infinite;}.loader-food.one {left: 15px; top: 30px; animation-delay: -.4s;}.loader-food.two {right: 15px; top: 24px; animation-delay: -.9s;}.loader-food.three {left: 47px; top: 2px; animation-delay: -1.3s;}.loader-food.four {right: 48px; top: 0; animation-delay: -1.7s;}
.loader-card h2 {font-size: 1.35rem; margin: 0 0 .4rem;}.loader-card p {color: #6b7280; margin: 0;}
.loader-progress {height: 5px; max-width: 205px; margin: 1.35rem auto 0; overflow: hidden; border-radius: 999px; background: #dbeee0;}.loader-progress span {display: block; width: 45%; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #2f855a, #8dcf89); animation: meal-progress 1.45s ease-in-out infinite;}
.recipe-link {white-space: nowrap;}
.account-identity {margin: .25rem 0 .8rem; line-height: 1.15;}
.account-identity span {display: block; color: #64748b; font-size: .78rem; letter-spacing: .06em; text-transform: uppercase;}
.account-identity strong {display: block; margin-top: .18rem; color: #1e293b; font-size: 1.08rem; font-weight: 700;}
.demo-access {margin: .85rem 0 .7rem; padding: .85rem .95rem; border: 1px solid #dbe8df; border-radius: 12px; background: #f7fbf8;}
.demo-access__label {font-size: .76rem; font-weight: 700; letter-spacing: .08em; color: #527562;}
.demo-access__copy {margin-top: .32rem; color: #475569; font-size: 1.05rem; line-height: 1.7;}.demo-access code {color: #1f7a4f; font-size: 1.08rem; font-weight: 700;}
.signup-prompt {margin: .7rem 0 .35rem; color: #64748b; font-size: .86rem;}
[data-testid="stButton"] > button {min-height: 2.5rem;}
[data-testid="stHeaderActionElements"] {display: none !important;}
@media (max-width: 1160px) {.block-container {max-width: 100%; padding-left: 1.25rem; padding-right: 1.25rem;}}
@keyframes salad-bounce {50% {transform: translateX(-50%) translateY(-5px) rotate(2deg);}}
@keyframes ingredient-float {50% {transform: translateY(-9px) rotate(8deg);}}
@keyframes meal-progress {0% {transform: translateX(-115%);} 55%, 100% {transform: translateX(280%);}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_all():
    """Load application resources once per server process."""
    engine = get_engine()
    initialize_database(engine)
    recipes_df, reviews_df, popularity_df, svd_model, tfidf_matrix, recipe_id_to_index = load_recommendation_assets()
    return engine, recipes_df, reviews_df, popularity_df, svd_model, tfidf_matrix, recipe_id_to_index


startup_overlay = render_loading_overlay("Preparing your meal studio", "Gathering everything for a smooth start…")
engine, recipes_df, reviews_df, popularity_df, svd_model, tfidf_matrix, recipe_id_to_index = load_all()
startup_overlay.empty()
recipe_names = recipes_df.drop_duplicates("RecipeId").set_index("RecipeId")["Name"].to_dict()


DEFAULTS = {
    "authenticated": False, "user_id": None, "page": "profile", "profile": None,
    "targets": None, "remaining_targets": None, "goal_label": None, "diet_type": "Non-Vegetarian",
    "meal_count": 3, "planner_mode": "Interactive", "recommendation_count": 5,
    "meal_step": 1, "meal_plan": [], "hybrid_static_df": None,
    "meal_recommendations": None, "meal_history": {}, "revisiting_meal_step": None,
    "restore_selected_step": None,
    "invalidated_recipe_ids": [],
    "is_generating_recommendations": False,
    "active_view": "planner", "auth_view": "login",
    "manual_target_candidate": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_plan():
    """Clear only the current meal-planning session."""
    st.session_state.page = "profile"
    st.session_state.profile = None
    st.session_state.targets = None
    st.session_state.remaining_targets = None
    st.session_state.goal_label = None
    st.session_state.meal_step = 1
    st.session_state.meal_plan = []
    st.session_state.hybrid_static_df = None
    st.session_state.meal_recommendations = None
    st.session_state.meal_history = {}
    st.session_state.revisiting_meal_step = None
    st.session_state.restore_selected_step = None
    st.session_state.invalidated_recipe_ids = []
    st.session_state.manual_target_candidate = None
    st.session_state.is_generating_recommendations = False


def go_back():
    """Return to the preceding planning state without discarding the full session."""
    if st.session_state.is_generating_recommendations:
        return
    if st.session_state.active_view != "planner":
        st.session_state.active_view = "planner"
        st.rerun()

    if st.session_state.page == "targets":
        st.session_state.page = "profile"
        st.session_state.manual_target_candidate = None
    elif st.session_state.page == "planner":
        if st.session_state.meal_plan:
            previous_recipe = st.session_state.meal_plan.pop()
            for macro in ("Calories", "Protein", "Carbs", "Fat"):
                st.session_state.remaining_targets[macro] += previous_recipe[f"Recommended{macro}"]
            st.session_state.meal_step = max(1, st.session_state.meal_step - 1)
            previous_step = st.session_state.meal_step
            st.session_state.meal_recommendations = st.session_state.meal_history.get(previous_step, {}).get("recommendations")
            st.session_state.revisiting_meal_step = previous_step
            st.session_state.restore_selected_step = previous_step
        else:
            st.session_state.page = "targets"
            st.session_state.hybrid_static_df = None
            st.session_state.meal_recommendations = None
            st.session_state.meal_history = {}
            st.session_state.revisiting_meal_step = None
            st.session_state.restore_selected_step = None
            st.session_state.invalidated_recipe_ids = []
    st.rerun()


def preference_sidebar():
    """Apply the view requested by the shared cookbook navigation component."""
    requested_view = render_preference_navigation(
        st.session_state.active_view,
        is_busy=st.session_state.is_generating_recommendations,
    )
    if requested_view:
        st.session_state.active_view = requested_view
        st.rerun()


def rated_recipes_screen():
    ratings = get_user_ratings(engine, st.session_state.user_id)
    edits, removals, save_changes = render_rated_recipe_editor(ratings, recipe_names)
    if save_changes:
        for recipe_id, rating in edits.items():
            if removals[recipe_id] or rating == 0:
                delete_rating(engine, st.session_state.user_id, recipe_id)
            else:
                upsert_rating(engine, st.session_state.user_id, recipe_id, rating)
        st.rerun()


def blocked_recipes_screen():
    restorations, restore_selected = render_blocked_recipe_editor(get_blocked_ids(engine, st.session_state.user_id), recipe_names)
    if restore_selected:
        for recipe_id, should_restore in restorations.items():
            if should_restore:
                unblock_recipe(engine, st.session_state.user_id, recipe_id)
        st.rerun()


def show_brand():
    render_brand("NutriCore", "Where data becomes diet")


def login_screen():
    show_brand()
    _, account_column, _ = st.columns([1, 1.35, 1])
    with account_column:
        st.subheader("Welcome back")
        user_id = st.text_input("User ID", autocomplete="username")
        passcode = st.text_input("Passcode", type="password", autocomplete="current-password")
        if st.button("Log in", type="primary", use_container_width=True):
            if authenticate_user(engine, user_id, passcode):
                st.session_state.authenticated = True
                st.session_state.user_id = user_id
                reset_plan()
                st.rerun()
            st.error("That User ID or passcode does not match.")
        st.markdown(
            """<div class="demo-access"><div class="demo-access__label">DEMO ACCESS</div>
            <div class="demo-access__copy">Passcode: <code>0000</code><br>Try IDs: <code>32143</code>, <code>8526</code>, <code>1533</code>, <code>12657</code>, <code>36944</code></div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<p class="signup-prompt">New here? Create a profile to save your own ratings and preferences.</p>', unsafe_allow_html=True)
        if st.button("Create an account", use_container_width=True):
            st.session_state.auth_view = "create_account"
            st.rerun()


def create_account_screen():
    show_brand()
    _, account_column, _ = st.columns([1, 1.35, 1])
    with account_column:
        st.subheader("Create an account")
        st.caption("Set up your NutriCore profile to save ratings and recipe preferences.")
        new_user_id = st.text_input("Choose a User ID", autocomplete="username")
        new_passcode = st.text_input("Choose a passcode", type="password", autocomplete="new-password")
        confirm_passcode = st.text_input("Confirm passcode", type="password", autocomplete="new-password")
        if st.button("Create account", type="primary", use_container_width=True):
            if not new_user_id.strip() or not new_passcode:
                st.error("Please choose both a User ID and passcode.")
            elif new_passcode != confirm_passcode:
                st.error("The passcodes do not match.")
            elif create_user(engine, new_user_id.strip(), new_passcode):
                st.success("Account created. You can log in now.")
                st.session_state.auth_view = "login"
            else:
                st.error("That User ID is already in use.")
        if st.button("Back to login", use_container_width=True):
            st.session_state.auth_view = "login"
            st.rerun()


def profile_screen():
    st.title("Let’s build your targets")
    st.caption("A few details are enough to suggest a daily calorie and macro target.")
    planner_mode = st.radio("Planning style", ["Interactive", "Automatic"], horizontal=True, key="profile_planner_mode")
    first, second = st.columns(2)
    with first:
        age = st.number_input("Age", min_value=16, max_value=100, value=25)
        gender = st.selectbox("Sex", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", min_value=35.0, max_value=300.0, value=70.0, step=0.5)
    with second:
        height = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=170.0, step=0.5)
        activity = st.selectbox("Activity level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Athlete"], index=2)
        diet_type = st.selectbox("Diet type", ["Non-Vegetarian", "Vegetarian", "Eggitarian"])
    meal_count = st.number_input("Meals per day", min_value=1, value=3, step=1)
    recommendation_count = st.session_state.recommendation_count
    if planner_mode == "Interactive":
        recommendation_count = st.number_input(
            "Recommendations to show for each interactive meal",
            min_value=1,
            max_value=20,
            value=st.session_state.recommendation_count,
            step=1,
        )
    if st.button("See my targets", type="primary"):
        bmr = calculate_bmr(age, gender, weight, height)
        st.session_state.profile = {"weight": weight, "tdee": calculate_tdee(bmr, activity)}
        st.session_state.diet_type = diet_type
        st.session_state.meal_count = meal_count
        st.session_state.planner_mode = planner_mode
        st.session_state.recommendation_count = recommendation_count
        st.session_state.page = "targets"
        st.rerun()


def start_planning(targets, goal_label):
    st.session_state.targets = targets.copy()
    st.session_state.remaining_targets = targets.copy()
    st.session_state.goal_label = goal_label
    st.session_state.meal_step = 1
    st.session_state.meal_plan = []
    st.session_state.meal_recommendations = None
    st.session_state.meal_history = {}
    st.session_state.revisiting_meal_step = None
    st.session_state.restore_selected_step = None
    st.session_state.invalidated_recipe_ids = []
    st.session_state.is_generating_recommendations = True
    loading_overlay = render_loading_overlay("Creating your plan", "Mixing your goals into the best meal ideas…")
    try:
        hybrid_static_df, _, _ = recommender.build_hybrid_static_df(
            user_id=st.session_state.user_id, recipes_df=recipes_df, reviews_df=reviews_df,
            pop_df=popularity_df, svd_model=svd_model, tfidf_matrix=tfidf_matrix,
            recipe_id_to_index=recipe_id_to_index,
        )
    finally:
        loading_overlay.empty()
        st.session_state.is_generating_recommendations = False
    st.session_state.hybrid_static_df = hybrid_static_df
    st.session_state.page = "planner"
    st.rerun()


def targets_screen():
    st.title("Choose your daily target")
    profiles = generate_all_goal_profiles(st.session_state.profile["weight"], generate_calorie_targets(st.session_state.profile["tdee"]))
    columns = st.columns(2)
    for index, row in profiles.iterrows():
        targets = {"Calories": float(row["Calories"]), "Protein": float(row["Protein_g"]), "Carbs": float(row["Carbs_g"]), "Fat": float(row["Fat_g"])}
        if len(profiles) % 2 and index == len(profiles) - 1:
            _, target_column, _ = st.columns([1, 2, 1])
        else:
            target_column = columns[index % 2]
        with target_column:
            with st.container(border=True):
                st.subheader(row["Goal"])
                render_macro_columns(targets)
                if st.button("Choose this target", key=f"goal_{index}", use_container_width=True):
                    start_planning(targets, row["Goal"])
    st.divider()
    st.subheader("Or set your own target")
    st.caption("Calories and protein are mandatory. Use whole numbers only; leave fat or carbs blank to calculate the missing macro.")
    first, second = st.columns(2)
    with first:
        calories = st.number_input("Calories — required", min_value=1, value=None, step=1, placeholder="e.g. 2000")
        protein = st.number_input("Protein (g) — required", min_value=1, value=None, step=1, placeholder="e.g. 140")
    with second:
        carbs = st.number_input("Carbs (g) — optional", min_value=1, value=None, step=1, placeholder="Calculated if blank")
        fat = st.number_input("Fat (g) — optional", min_value=1, value=None, step=1, placeholder="Calculated if blank")
    review_manual_target = st.button("Review manual target")
    if review_manual_target:
        if calories is None or protein is None:
            st.session_state.manual_target_candidate = None
            st.error("Calories and protein are required before reviewing a manual target.")
        else:
            calories, protein = parse_optional_whole_number(calories), parse_optional_whole_number(protein)
            carbs, fat = parse_optional_whole_number(carbs), parse_optional_whole_number(fat)
            targets = generate_manual_targets(calories, protein, fat=fat, carbs=carbs)
            formatted = {
                "Calories": float(targets["Calories"]),
                "Protein": float(targets["Protein_g"]),
                "Carbs": float(targets["Carbs_g"]),
                "Fat": float(targets["Fat_g"]),
            }
            if any(value <= 0 for value in formatted.values()):
                st.session_state.manual_target_candidate = None
                st.error("All final calorie and macro targets must be greater than zero.")
            else:
                st.session_state.manual_target_candidate = {
                    "targets": formatted,
                    "calculated_macros": [macro for macro, value in (("Carbs", carbs), ("Fat", fat)) if value is None],
                    "validation": validate_macros(calories, protein, formatted["Fat"], formatted["Carbs"]),
                    "quality": assess_macro_quality(calories, protein, formatted["Fat"], formatted["Carbs"]),
                    "eligibility": check_recommendation_eligibility(calories, protein, formatted["Fat"], formatted["Carbs"]),
                }

    candidate = st.session_state.manual_target_candidate
    if candidate:
        st.divider()
        st.subheader("Confirm your manual target")
        render_macro_columns(candidate["targets"])
        if candidate["calculated_macros"]:
            st.info(f"NutriCore calculated {' and '.join(candidate['calculated_macros']).lower()} from the values you provided.")
        validation = candidate["validation"]
        if validation["Valid"]:
            st.success(f"Macro calories are aligned: {validation['Macro Calories']:.0f} kcal against a {validation['Target Calories']:.0f} kcal target.")
        else:
            st.warning(f"Your macros add up to {validation['Macro Calories']:.0f} kcal, which is {validation['Difference']:.0f} kcal away from the calorie target.")
        for warning in candidate["quality"]["Warnings"]:
            if warning != "No Warnings":
                st.warning(warning)
        for recommendation in candidate["quality"]["Recommendations"]:
            st.info(recommendation)
        eligibility = candidate["eligibility"]
        if not eligibility["Eligible"]:
            st.error(" ".join(eligibility["Reasons"]))
        if st.button("Use these targets", type="primary", disabled=not eligibility["Eligible"]):
            start_planning(candidate["targets"], "Manual target")


def invalidate_later_meals(current_step):
    """Keep earlier choices, but remove suggestions based on a replaced meal."""
    history = st.session_state.meal_history
    for step in [step for step in history if step > current_step]:
        st.session_state.invalidated_recipe_ids.extend(history[step]["recommendations"]["RecipeId"].tolist())
        del history[step]


def save_current_meal(results, choices):
    selected_ids = [recipe_id for recipe_id, choice in choices.items() if choice["selected"]]
    if len(selected_ids) != 1:
        st.error("Choose one meal card before continuing.")
        return

    selected_id = selected_ids[0]
    current_step = st.session_state.meal_step
    history_entry = st.session_state.meal_history.setdefault(current_step, {})
    history_entry["recommendations"] = results.copy()
    previous_selection = history_entry.get("selected_id")
    if previous_selection is not None and previous_selection != selected_id:
        invalidate_later_meals(current_step)
    history_entry["selected_id"] = selected_id
    for recipe_id, choice in choices.items():
        if choice["rating"] == 0:
            delete_rating(engine, st.session_state.user_id, recipe_id)
        else:
            upsert_rating(engine, st.session_state.user_id, recipe_id, choice["rating"])
        if choice["block"]:
            block_recipe(engine, st.session_state.user_id, recipe_id)

    recipe = choices[selected_id]["recipe"]
    st.session_state.meal_plan.append(recipe.to_dict())
    for macro in ("Calories", "Protein", "Carbs", "Fat"):
        st.session_state.remaining_targets[macro] -= recipe[f"Recommended{macro}"]
    st.session_state.meal_step += 1
    next_entry = st.session_state.meal_history.get(st.session_state.meal_step, {})
    st.session_state.meal_recommendations = next_entry.get("recommendations")
    st.session_state.revisiting_meal_step = None
    st.session_state.restore_selected_step = (
        st.session_state.meal_step
        if next_entry.get("selected_id") is not None and st.session_state.meal_recommendations is not None
        else None
    )
    st.session_state.is_generating_recommendations = (
        st.session_state.meal_step <= st.session_state.meal_count
        and st.session_state.meal_recommendations is None
    )
    st.rerun()


def planner_screen():
    if st.session_state.meal_step > st.session_state.meal_count:
        finish_screen()
        return
    step = st.session_state.meal_step
    st.title(f"Meal {step} of {st.session_state.meal_count}")
    if st.session_state.revisiting_meal_step == step:
        st.info("You are revisiting this meal. Your earlier choice is preselected. Keeping it preserves every later meal; replacing it refreshes only the meals after this one.")
    remaining_meals = st.session_state.meal_count - step + 1
    results = st.session_state.meal_recommendations
    if results is None:
        targets = st.session_state.remaining_targets
        blocked_ids = get_blocked_ids(engine, st.session_state.user_id)
        used_ids = [recipe["RecipeId"] for recipe in st.session_state.meal_plan]
        top_n = 1 if st.session_state.planner_mode == "Automatic" else st.session_state.recommendation_count
        st.session_state.is_generating_recommendations = True
        loading_overlay = render_loading_overlay("Plating your next meal", "Finding options that match your goals…")
        try:
            results = recommender.generate_hybrid_recommendations(
                recipes_df=recipes_df, hybrid_static_df=st.session_state.hybrid_static_df,
                calories=targets["Calories"], protein=targets["Protein"], carbs=targets["Carbs"], fat=targets["Fat"],
                meals_per_day=remaining_meals, diet_type=st.session_state.diet_type,
                top_n=top_n + len(blocked_ids) + len(used_ids),
            )
            results = results[~results["RecipeId"].isin(blocked_ids) & ~results["RecipeId"].isin(used_ids)].head(top_n)
        finally:
            loading_overlay.empty()
            st.session_state.is_generating_recommendations = False
        st.session_state.meal_recommendations = results
        history_entry = st.session_state.meal_history.setdefault(step, {})
        history_entry["recommendations"] = results.copy()
    if results.empty:
        st.warning("There are no available recipes for these remaining targets. Start a new plan to adjust them.")
        if st.button("Start over"):
            reset_plan()
            st.rerun()
        return
    for recipe_id in set(st.session_state.invalidated_recipe_ids):
        for prefix in ("select_", "block_", "rating_"):
            st.session_state.pop(f"{prefix}{recipe_id}", None)
    st.session_state.invalidated_recipe_ids = []
    if st.session_state.restore_selected_step == step:
        restore_selected_recipe(results, st.session_state.meal_history.get(step, {}).get("selected_id"))
        st.session_state.restore_selected_step = None
    if step == st.session_state.meal_count:
        render_final_meal_hint(results.iloc[0])
    render_meal_cards(
        results, get_user_ratings(engine, st.session_state.user_id), st.session_state.planner_mode == "Interactive",
        step, st.session_state.meal_count, save_current_meal,
    )


def finish_screen():
    st.title("Your plan is ready")
    plan = pd.DataFrame(st.session_state.meal_plan)
    totals = plan_totals(plan)
    render_plan_summary(st.session_state.targets, totals, st.session_state.goal_label or "Selected target")

    total_difference = sum(abs(totals[macro] - st.session_state.targets[macro]) for macro in totals)
    if total_difference < 60:
        st.success("✨ A neatly balanced plate parade — your plan is very close to the targets you chose.")
    else:
        st.info("✨ Your meal map is ready. The tally above shows exactly how today’s plates compare with your target.")

    render_plan_meals(st.session_state.meal_plan)
    if st.button("Build another plan", type="primary"):
        reset_plan()
        st.rerun()


def application():
    preference_sidebar()
    with st.container():
        st.markdown('<div style="height: .5rem"></div>', unsafe_allow_html=True)
        can_go_back = st.session_state.active_view != "planner" or st.session_state.page != "profile"
        identity_column, logout_column = st.columns([5, 1], vertical_alignment="top")
        with identity_column:
            safe_user_id = html.escape(str(st.session_state.user_id))
            st.markdown(
                f'<div class="account-identity"><span>Signed in as</span><strong>{safe_user_id}</strong></div>',
                unsafe_allow_html=True,
            )
            if can_go_back and st.button("← Back", disabled=st.session_state.is_generating_recommendations):
                go_back()
        with logout_column:
            if st.button("Log out", use_container_width=True, disabled=st.session_state.is_generating_recommendations):
                st.session_state.clear()
                st.rerun()
        st.divider()
    if st.session_state.active_view == "rated_recipes":
        rated_recipes_screen()
        return
    if st.session_state.active_view == "blocked_recipes":
        blocked_recipes_screen()
        return
    if st.session_state.page == "profile":
        profile_screen()
    elif st.session_state.page == "targets":
        targets_screen()
    else:
        planner_screen()


if st.session_state.authenticated:
    application()
elif st.session_state.auth_view == "create_account":
    create_account_screen()
else:
    login_screen()
