# NutriCore

*Where data becomes diet.*

NutriCore is a Streamlit meal-planning application built on Food.com recipe data. It calculates nutrition targets, produces hybrid recommendations, lets users refine meal choices interactively, and preserves recipe ratings and blocked recipes in a local database.

## Repository layout

```
nutricore/
├── app.py                         # Streamlit entry point
├── data/
│   ├── raw/                       # Original Food.com extracts
│   └── processed/                 # Runtime recommendation datasets
├── models/                        # Pretrained SVD and TF-IDF artifacts
├── notebooks/                     # EDA, modelling, and planner experiments
│   └── nb_utils/                  # Notebook-only helper modules
└── src/                           # Production application modules
```

`src/ui.py` contains reusable Streamlit components in addition to the modules listed above; it keeps page layout and rendering separate from the planning workflow in `app.py`.

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app from the repository root:

   ```bash
   streamlit run app.py
   ```

The repository includes its SQLite database, `nutricore.db`, at the root so the bundled demo profiles work immediately. To use a different database, set `DATABASE_URL`, for example `sqlite:///my_nutricore.db`. Run `python setup_db.py` only when you need to build a database from the processed reviews dataset.

## Data and model artifacts

The repository layout includes raw data, processed recommendation inputs, and trained model files. Before pushing these large binaries to GitHub, configure Git LFS or host the artifacts externally and document how to retrieve them.
