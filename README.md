# Mental Health in Tech — Analytics & Prediction (Streamlit)

A recruiter-facing, interactive deployment of the *Mental Health in Tech Survey EDA & ML*
capstone, now in a dark, futuristic theme. Built on the real 2014 OSMI survey (1,259
respondents) plus the 2016 companion wave, with the same cleaning pipeline as the source
notebook, expanded to **six models**: Logistic Regression, Random Forest, Gradient Boosting,
Support Vector Machine, K-Nearest Neighbors, and a small Neural Network (MLP).

## What's inside

```
mh_app/
├── app.py                  # the whole Streamlit app (single file)
├── requirements.txt
├── .streamlit/config.toml  # theme
└── data/
    ├── survey.csv                              # 2014 OSMI survey (1259 x 27)
    └── mental-heath-in-tech-2016_20161114.csv   # 2016 OSMI wave (1433 x 63)
```

Pages: **Overview** (KPIs + story) · **Explore & Segment** (filterable demographics) ·
**Correlations & Drivers** (heatmap + policy-vs-treatment charts) · **Model Performance**
(radar chart across all six models, a model picker for ROC/confusion matrix/report, feature
importance) · **Try the Prediction Tool** (pick which of the six models drives the live
verdict, compare all six side by side) · **2014 vs 2016 Trends** · **Recommendations & About**.

Models are trained once at startup (cached with `@st.cache_resource`) directly from the CSVs —
no `.pkl` files needed, so there's nothing to keep in sync with the notebook. No new
dependencies were needed for the add-on models — SVM, KNN, and MLP all come from the
scikit-learn install already in `requirements.txt`.

Theme: a dark, glassmorphism "futuristic" look — animated gradient hero, neon cyan/purple/pink
accents, glowing KPI cards — driven by `.streamlit/config.toml` plus custom CSS in `app.py`,
with a matching dark Plotly template applied to every chart.

## Run locally

```bash
cd mh_app
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploy for free — Streamlit Community Cloud

1. Push this folder to a public (or private) GitHub repo, keeping the `data/` folder and
   `.streamlit/config.toml` in place.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo, branch, and set the main file path to `app.py`.
4. Deploy. First load will take ~30–60s while it installs dependencies and trains the models;
   subsequent loads are instant thanks to caching.
5. Copy the generated `*.streamlit.app` URL into your resume/portfolio/LinkedIn.

## Deploy alternatives

- **Hugging Face Spaces** (Streamlit SDK): create a Space, upload the same files, it builds
  automatically from `requirements.txt`.
- **Render / Railway**: use a simple start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## Notes

- The app pulls the real OSMI datasets bundled in `data/` — no external network calls at runtime.
- If you swap in your own survey export, keep the same column names used in `app.py`'s
  `ENCODE_COLS` / cleaning function, or adjust the cleaning function to match your schema.
