# Repository Guidelines

## Project Structure & Module Organization
- `app.py` is the Streamlit application and contains all business logic and UI.
- `data/` holds example CSV inputs (historical totals, deliveries, baseline data).
- `.fuel_profit_data/` is the local runtime data store created by the app (generated on first run).
- `.venv/` is an optional local virtual environment (not required by the repo, but present here).
- `requirements.txt` lists core Python dependencies for the app.

## Build, Test, and Development Commands
- `python -m venv .venv` to create a virtual environment (recommended).
- `source .venv/bin/activate` to activate the venv.
- `python -m pip install -r requirements.txt` to install core dependencies.
- `python -m pip install streamlit-aggrid` to enable the optional AgGrid UI.
- `streamlit run app.py` to launch the app locally.

## Coding Style & Naming Conventions
- Python, 4-space indentation, PEP 8-style layout.
- `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants.
- Keep data transforms in small, testable functions; keep Streamlit UI wiring in the main flow.
- No formatter or linter is configured; keep diffs clean and avoid unused imports.

## Testing Guidelines
- No automated test framework is configured.
- Validate changes by running the app and exercising key flows:
  - Load CSVs from `data/`.
  - Check daily totals, tank math, and profit summaries for expected values.
- If you add tests, document how to run them in this file.

## Commit & Pull Request Guidelines
- No Git history is present in this repository, so there are no established conventions.
- If you add Git usage, keep commits small and descriptive (e.g., “Fix tank delivery aggregation”).
- For PRs, include:
  - A brief summary of behavior changes.
  - Steps to verify (commands + datasets).
  - Screenshots for UI changes when relevant.

## Configuration & Data Safety
- `.fuel_profit_data/` is user-specific; avoid committing real operational data.
- Prefer using `data/` samples when sharing or reproducing issues.
