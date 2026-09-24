"""Reusable Streamlit presentation helpers."""

import streamlit as st

from src.utils import MACROS, display_number, generate_foodcom_url, preference_recipe_url, recipe_name


RATING_OPTIONS = [step / 4 for step in range(21)]


def rating_stars(value: float) -> str:
    """Format quarter-star values for Streamlit's rating selector."""
    full_stars = int(value)
    fraction = round((value - full_stars) * 4)
    partial_star = {0: "", 1: "¼", 2: "½", 3: "¾"}[fraction]
    empty_stars = 5 - full_stars - (1 if fraction else 0)
    return "★" * full_stars + partial_star + "☆" * empty_stars


def render_loading_overlay(title: str, subtitle: str):
    """Show an interaction-blocking loading overlay and return its placeholder."""
    placeholder = st.empty()
    placeholder.markdown(
        f"""<div class="loader-overlay" role="status" aria-live="assertive"><div class="loader-card">
        <div class="loader-scene"><span class="loader-food one">🍎</span><span class="loader-food two">🥑</span>
        <span class="loader-food three">🥕</span><span class="loader-food four">🥦</span>
        <div class="loader-plate"></div><div class="loader-salad"></div></div>
        <h2>{title}</h2><p>{subtitle}</p><div class="loader-progress"><span></span></div></div></div>""",
        unsafe_allow_html=True,
    )
    return placeholder


def render_brand(name: str, tagline: str) -> None:
    """Render the product name and tagline on unauthenticated screens."""
    st.markdown(
        f'<div class="hero"><h1>🥗 {name}</h1><p><em>{tagline}</em></p></div>',
        unsafe_allow_html=True,
    )


def render_macro_columns(targets: dict[str, float]) -> None:
    """Render calories and macro targets in a consistent four-column layout."""
    values = (
        ("Calories", f"{round(targets['Calories'])} kcal"),
        ("Protein", f"{round(targets['Protein'])} g"),
        ("Carbs", f"{round(targets['Carbs'])} g"),
        ("Fat", f"{round(targets['Fat'])} g"),
    )
    for column, (label, value) in zip(st.columns(4), values):
        with column:
            st.markdown(
                f'<p class="macro-label">{label}</p><p class="macro-value">{value}</p>',
                unsafe_allow_html=True,
            )


def render_preference_navigation(active_view: str, is_busy: bool = False) -> str | None:
    """Render cookbook navigation and return the requested destination, if any."""
    with st.sidebar:
        st.header("Your cookbook")
        if is_busy:
            st.caption("Recommendations are updating. Your cookbook will be ready in a moment.")
        else:
            st.caption("Your meal plan stays exactly where it is while you manage preferences.")
        for label, view in (
            ("Meal planner", "planner"),
            ("Rated recipes", "rated_recipes"),
            ("Blocked recipes", "blocked_recipes"),
        ):
            if st.button(
                label,
                key=f"sidebar_{view}",
                use_container_width=True,
                disabled=is_busy or active_view == view,
            ):
                return view
    return None


def render_rated_recipe_editor(ratings: dict, recipe_names: dict) -> tuple[dict, dict, bool]:
    """Render the concise rated-recipe table and return pending edits."""
    rated_ids = sorted(
        (recipe_id for recipe_id, rating in ratings.items() if rating > 0),
        key=lambda recipe_id: recipe_name(recipe_id, recipe_names).lower(),
    )
    st.title("Rated recipes")
    st.caption("Edit ratings or remove saved ratings. Your meal plan is preserved in the Meal planner view.")
    if not rated_ids:
        st.info("Recipes you rate will appear here.")
        return {}, {}, False

    st.caption(f"{len(rated_ids)} rated recipe{'s' if len(rated_ids) != 1 else ''}")
    header = st.columns([3.3, 2.6, 1.2, 1])
    for column, label in zip(header, ("RECIPE", "RATING", "LINK", "REMOVE")):
        column.caption(label)
    edits, removals = {}, {}
    for recipe_id in rated_ids:
        row = st.columns([3.3, 2.6, 1.2, 1])
        with row[0]:
            st.write(recipe_name(recipe_id, recipe_names))
        with row[1]:
            edits[recipe_id] = st.select_slider(
                "Rating", options=RATING_OPTIONS, value=float(ratings[recipe_id]),
                format_func=rating_stars, label_visibility="collapsed", key=f"manage_rating_{recipe_id}",
            )
        with row[2]:
            st.markdown(f"[Food.com ↗]({preference_recipe_url(recipe_id, recipe_names)})")
        with row[3]:
            removals[recipe_id] = st.checkbox("Remove", label_visibility="collapsed", key=f"remove_rating_{recipe_id}")
    return edits, removals, st.button("Save rating changes", type="primary")


def render_blocked_recipe_editor(blocked_ids: set, recipe_names: dict) -> tuple[dict, bool]:
    """Render the concise blocked-recipe table and return requested restorations."""
    blocked_ids = sorted(blocked_ids, key=lambda recipe_id: recipe_name(recipe_id, recipe_names).lower())
    st.title("Blocked recipes")
    st.caption("Choose any recipes to restore to future recommendations. Your meal plan is preserved in the Meal planner view.")
    if not blocked_ids:
        st.info("Recipes you block while planning will appear here.")
        return {}, False

    st.caption(f"{len(blocked_ids)} blocked recipe{'s' if len(blocked_ids) != 1 else ''}")
    header = st.columns([4.6, 1.5, 1])
    for column, label in zip(header, ("RECIPE", "LINK", "RESTORE")):
        column.caption(label)
    restorations = {}
    for recipe_id in blocked_ids:
        row = st.columns([4.6, 1.5, 1])
        with row[0]:
            st.write(recipe_name(recipe_id, recipe_names))
        with row[1]:
            st.markdown(f"[Food.com ↗]({preference_recipe_url(recipe_id, recipe_names)})")
        with row[2]:
            restorations[recipe_id] = st.checkbox("Restore", label_visibility="collapsed", key=f"restore_block_{recipe_id}")
    return restorations, st.button("Restore selected recipes", type="primary")


def render_recipe_card(recipe, existing_rating: float, show_selection: bool, recipe_ids: list, on_selection_changed) -> tuple[float, bool, bool]:
    """Render one recommendation card and return its rating, block, and selection state."""
    with st.container(border=True):
        st.subheader(recipe["Name"])
        render_macro_columns({macro: recipe[f"Recommended{macro}"] for macro in MACROS})
        serving_display = display_number(recipe["BestServing"])
        st.caption(f"Recommended serving size: {serving_display}")
        rating_col, select_col, block_col, link_col = st.columns([1.25, 1, 1, 1.7])
        with rating_col:
            rating = st.select_slider(
                "Your rating", options=RATING_OPTIONS, value=float(existing_rating),
                format_func=rating_stars, key=f"rating_{recipe['RecipeId']}",
            )
        with select_col:
            selected = st.checkbox(
                "Select this meal", key=f"select_{recipe['RecipeId']}", on_change=on_selection_changed,
                args=(recipe["RecipeId"], recipe_ids),
            ) if show_selection else True
        with block_col:
            blocked = st.checkbox("Add to block list", key=f"block_{recipe['RecipeId']}")
        with link_col:
            st.markdown(f"[View recipe on Food.com ↗]({generate_foodcom_url(recipe['Name'], recipe['RecipeId'], serving_display)})")
    return rating, blocked, selected


def keep_one_recipe_selected(recipe_id, recipe_ids) -> None:
    """Keep interactive meal-card selection mutually exclusive."""
    if st.session_state.get(f"select_{recipe_id}"):
        for other_id in recipe_ids:
            if other_id != recipe_id:
                st.session_state[f"select_{other_id}"] = False


def restore_selected_recipe(results, selected_id) -> None:
    """Preselect a saved recipe before its recommendation cards are rendered."""
    if selected_id is not None:
        for recipe_id in results["RecipeId"].tolist():
            st.session_state[f"select_{recipe_id}"] = recipe_id == selected_id


@st.fragment
def render_meal_cards(results, ratings: dict, interactive: bool, meal_step: int, meal_count: int, on_continue) -> None:
    """Render recommendation cards and pass their choices to the planning workflow."""
    choices = {}
    recipe_ids = results["RecipeId"].tolist()
    for _, recipe in results.iterrows():
        rating, should_block, selected = render_recipe_card(
            recipe, ratings.get(recipe["RecipeId"], 0.0), interactive, recipe_ids, keep_one_recipe_selected,
        )
        choices[recipe["RecipeId"]] = {
            "recipe": recipe, "rating": rating, "block": should_block, "selected": selected,
        }
    action_label = "Finish meal planning" if meal_step == meal_count else "Next meal"
    if st.button(action_label, type="primary", key=f"next_meal_{meal_step}"):
        on_continue(results, choices)


def render_final_meal_hint(recipe) -> None:
    """Call out the highest-ranked final-meal option without preventing other choices."""
    st.success(f"Best fit for your final meal: **{recipe['Name']}**. It is the closest match for your remaining targets.")


def render_plan_summary(targets: dict, totals: dict, goal_label: str) -> None:
    """Render the final goal, target, planned-total, and rounded macro deltas."""
    st.subheader("Your target and planned total")
    st.caption(f"GOAL · {goal_label}")
    for column, macro in zip(st.columns(4), MACROS):
        target, planned = round(targets[macro]), round(totals[macro])
        unit = "kcal" if macro == "Calories" else "g"
        difference = planned - target
        with column:
            st.caption(f"Target: {target} {unit}")
            if difference == 0:
                st.metric(macro, f"{planned} {unit}", delta="On target ✓", delta_color="normal")
            else:
                st.metric(
                    macro, f"{planned} {unit}",
                    delta=f"{difference:+d} {unit} {'over' if difference > 0 else 'under'} target", delta_color="off",
                )


def render_plan_meals(meal_plan: list) -> None:
    """Render the selected recipes in the final plan."""
    st.subheader("Your meals")
    for meal_number, recipe in enumerate(meal_plan, start=1):
        with st.container(border=True):
            st.caption(f"MEAL {meal_number}")
            st.subheader(recipe["Name"])
            render_macro_columns({macro: recipe[f"Recommended{macro}"] for macro in MACROS})
            serving_display = display_number(recipe["BestServing"])
            st.caption(f"Recommended serving size: {serving_display}")
            st.markdown(f"[View recipe on Food.com ↗]({generate_foodcom_url(recipe['Name'], recipe['RecipeId'], serving_display)})")


__all__ = [
    "RATING_OPTIONS",
    "rating_stars",
    "render_brand",
    "render_blocked_recipe_editor",
    "render_final_meal_hint",
    "render_loading_overlay",
    "render_macro_columns",
    "render_meal_cards",
    "render_plan_meals",
    "render_plan_summary",
    "render_preference_navigation",
    "render_rated_recipe_editor",
    "render_recipe_card",
    "restore_selected_recipe",
]
