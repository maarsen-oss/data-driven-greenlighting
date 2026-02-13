import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ============================================================
# LOAD CLEANED DATASET
# ============================================================

df_clean = pd.read_csv("TMDB_movie_dataset_v11_clean.csv")

# ============================================================
# FEATURE ENGINEERING (UPGRADED MODEL)
# ============================================================

genres = ['Mystery', 'Horror', 'Thriller', 'Drama', 'Crime']

# Genre indicators
for g in genres:
    df_clean[f'is_{g.lower()}'] = df_clean['genres'].str.contains(g, na=False).astype(int)

# Success variable
df_clean['success'] = (df_clean['roi'] > 1.5).astype(int)
# Budget tier
def budget_tier(b):
    if b < 1_000_000: return 0
    elif b < 10_000_000: return 1
    elif b < 40_000_000: return 2
    elif b < 100_000_000: return 3
    else: return 4

df_clean['budget_tier'] = df_clean['budget'].apply(budget_tier)

# Runtime tier
def runtime_tier(r):
    if r < 85: return 0
    elif r < 105: return 1
    elif r < 130: return 2
    else: return 3

df_clean['runtime_tier'] = df_clean['runtime'].apply(runtime_tier)

# Genre count + multigenre
df_clean['genre_count'] = df_clean['genres'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
df_clean['is_multigenre'] = (df_clean['genre_count'] > 1).astype(int)

# Nonlinear transforms
df_clean['log_budget'] = np.log1p(df_clean['budget'])
df_clean['log_runtime'] = np.log1p(df_clean['runtime'])
df_clean['budget_tier_sq'] = df_clean['budget_tier'] ** 2
df_clean['runtime_tier_sq'] = df_clean['runtime_tier'] ** 2

# Interaction terms
for g in genres:
    df_clean[f'{g.lower()}_x_budget'] = df_clean[f'is_{g.lower()}'] * df_clean['budget_tier']
    df_clean[f'{g.lower()}_x_runtime'] = df_clean[f'is_{g.lower()}'] * df_clean['runtime_tier']

df_clean['runtime_x_budget'] = df_clean['runtime_tier'] * df_clean['budget_tier']

# Release season
df_clean['release_month'] = pd.to_datetime(df_clean['release_date'], errors='coerce').dt.month
df_clean['season'] = pd.cut(df_clean['release_month'], bins=[0,3,6,9,12], labels=['Q1','Q2','Q3','Q4'])

df_clean['is_summer'] = df_clean['release_month'].isin([5,6,7,8]).astype(int)
df_clean['is_holiday'] = df_clean['release_month'].isin([11,12]).astype(int)

df_clean = pd.get_dummies(df_clean, columns=['season'], drop_first=True)

# Budget efficiency
df_clean['budget_per_genre'] = df_clean['budget'] / (df_clean['genre_count'] + 1)

# ============================================================
# MODEL FEATURES
# ===========================================
feature_cols = (
    [
        'log_budget', 'log_runtime',
        'budget_tier', 'budget_tier_sq',
        'runtime_tier', 'runtime_tier_sq',
        'genre_count', 'is_multigenre',
        'budget_per_genre',
        'runtime_x_budget'
    ]
    + [f'is_{g.lower()}' for g in genres]
    + [f'{g.lower()}_x_budget' for g in genres]
    + [f'{g.lower()}_x_runtime' for g in genres]
    + ['is_summer', 'is_holiday']
    + [c for c in df_clean.columns if c.startswith('season_')]
)

X = df_clean[feature_cols]
y = df_clean['success']

# ============================================================
# TRAIN MODEL
# ============================================================

model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=2000))
])

model.fit(X, y)


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🎬 Movie Greenlight Decision Tool — Enhanced Model")
st.write("A data-driven logistic regression model using engineered features to evaluate movie greenlight decisions.")

# User inputs
budget = st.number_input("Budget ($)", min_value=10000, max_value=500_000_000, value=40_000_000)
runtime = st.number_input("Runtime (minutes)", min_value=60, max_value=240, value=110)

st.subheader("Genres")
genre_dict = {g: st.checkbox(g) for g in genres}

release_month = st.selectbox("Release Month", list(range(1,13)))

# ============================================================
# BUILD INPUT ROW (MATCHES IMPROVED MODEL)
row = pd.DataFrame([{
    'log_budget': np.log1p(budget),
    'log_runtime': np.log1p(runtime),
    'budget_tier': budget_tier(budget),
    'budget_tier_sq': budget_tier(budget)**2,
    'runtime_tier': runtime_tier(runtime),
    'runtime_tier_sq': runtime_tier(runtime)**2,
    'genre_count': sum(genre_dict.values()),
    'is_multigenre': int(sum(genre_dict.values()) > 1),
    'budget_per_genre': budget / (sum(genre_dict.values()) + 1),

    # MUST BE HERE — index 9
    'runtime_x_budget': runtime_tier(runtime) * budget_tier(budget),

    # genre indicators
    **{f'is_{g.lower()}': int(genre_dict[g]) for g in genres},

    # interactions
    **{f'{g.lower()}_x_budget': int(genre_dict[g]) * budget_tier(budget) for g in genres},
    **{f'{g.lower()}_x_runtime': int(genre_dict[g]) * runtime_tier(runtime) for g in genres},

    # seasonal flags
    'is_summer': int(release_month in [5,6,7,8]),
    'is_holiday': int(release_month in [11,12]),

    # season dummies
    'season_Q2': 1 if release_month in [4,5,6] else 0,
    'season_Q3': 1 if release_month in [7,8,9] else 0,
    'season_Q4': 1 if release_month in [10,11,12] else 0,
}])



# ================================
# PREDICT
# ============================================================

prob = model.predict_proba(row)[0][1]

if prob > 0.70:
    decision = f"YES — High likelihood of success ({prob:.2f})"
elif prob > 0.50:
    decision = f"YES WITH CAUTION — Moderate likelihood ({prob:.2f})"
else:
    decision = f"NO — Low likelihood ({prob:.2f})"

# ============================================================
# OUTPUT
# ============================================================

st.subheader("Prediction")
st.write(f"**Probability of Success:** {prob:.2f}")
st.write(f"**Decision:** {decision}")
