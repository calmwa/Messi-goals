# Messi Goals: Dimension Reduction

Interactive PCA and t-SNE exploration of the repository’s Messi goal records. The original notebooks and CSV are preserved.

## Setup

From the repository root with Python 3.10 or newer:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Execute the notebook

```bash
python -m nbconvert --execute --to notebook --inplace final_dimension_reduction.ipynb
```

The notebook loads data relative to the repository root and writes `goal_projections.csv` there.

## Run the dashboard

```bash
streamlit run app.py
```

Choose seasons, competitions, goal types, and venues in the sidebar. Select records by goal ID in the multiselect; the same IDs are highlighted in PCA, t-SNE, the original feature summaries, and the table. “Reset selection” clears the set. Download the currently filtered projection table with the download button.

Expand **Approved bonus: context projection matrix** to compare PCA subspaces for match, goal, and career context. Each of the nine cells can be enlarged with its button, and the active goal-ID selection remains highlighted throughout. See `bonus_extension.md` for the adaptation and its limits.

## Method notes

The mixed feature matrix standardizes minute and season-start year and one-hot encodes competition, venue, position, goal type, and opponent. Rare opponent levels (fewer than five examples) are pooled. Assists are excluded because 215 records are missing and 69 distinct names create high-cardinality identity sparsity. IDs, date, matchday, club, and score/result fields are excluded as identifiers, redundant fields, or outcome-like context. PCA and t-SNE use a fixed seed. Trustworthiness uses a shared valid neighborhood size and the t-SNE perplexity is always less than sample count.

## Rubric checklist

- [x] PCA and t-SNE on a common preprocessed matrix; variance, trustworthiness, reproducible parameters.
- [x] Original-space distributions and parallel coordinates, plus labeled 2D projections and computed comparison.
- [x] Linked selection by immutable `goal_id`, filters, hover details, original feature comparison, detail table, and reset.
- [x] Editable peer evaluation template; students submit their own evaluation separately.
- [x] Report sections on provenance, methodology, interpretation, limitations, conclusion, and reproducibility.
- [x] Tests for parsing, feature preparation, PCA/t-SNE shapes, ID alignment, and all nine bonus subspaces.
- [x] Approved bonus: interactive 3x3 PCA context projection matrix; see `bonus_extension.md` for method and simplifications.

## Validation commands

```bash
python -m pytest -q
python -m streamlit run app.py --server.headless true
```

Run the second command in a terminal and open the displayed local URL. Notebook execution command is listed above.
