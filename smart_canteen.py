"""
Smart Canteen: Machine Learning Based Food Demand Prediction to Reduce Food Waste
---------------------------------------------------------------------------------
Part 1 (machine learning): predict plates sold per day (Linear Regression vs Random Forest).
Part 2 (rule-based planner): turn the prediction into plates to prepare per meal slot.

NOTE: the data is SYNTHETIC (generated below). To use real data, replace the
"1. DATASET" block with  df = pd.read_csv("your_canteen_sales.csv")  using the same columns:
attendance_pct, temperature_c, menu_rating, exam_week, weekend, plates_sold
"""
import math
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- 1. DATASET (synthetic) ----------
rng = np.random.default_rng(42)
n = 600
df = pd.DataFrame({
    "attendance_pct": rng.uniform(40, 100, n).round(1),
    "temperature_c": rng.normal(30, 5, n).clip(15, 43).round(1),
    "menu_rating": rng.uniform(2, 5, n).round(1),
    "exam_week": rng.binomial(1, 0.2, n),
    "weekend": rng.binomial(1, 0.2, n),
})
plates = (25 + 3.4*df.attendance_pct + 18*(df.menu_rating-3)
          - 0.35*(df.temperature_c-28)**2 + 20*df.exam_week - 60*df.weekend
          + rng.normal(0, 15, n))
df["plates_sold"] = plates.clip(20, None).round().astype(int)
df.loc[rng.choice(n, 10, replace=False), "menu_rating"] = np.nan   # missing values
df = pd.concat([df, df.iloc[:5]], ignore_index=True)               # duplicates
df.to_csv("canteen_dataset_raw.csv", index=False)

print("=== DATASET HEAD ==="); print(df.head())
print("\n=== SHAPE ===", df.shape)
print("\n=== MISSING VALUES ==="); print(df.isna().sum())

# ---------- 2. PREPROCESSING ----------
df = df.drop_duplicates()
df["menu_rating"] = df["menu_rating"].fillna(df["menu_rating"].median())
print("\nAfter cleaning:", df.shape, "| missing =", int(df.isna().sum().sum()))
print("\n=== DESCRIPTIVE STATISTICS ==="); print(df.describe().round(2))
print("\n=== CORRELATION WITH PLATES SOLD ===")
print(df.corr()["plates_sold"].drop("plates_sold").round(3))

X, y = df.drop(columns="plates_sold"), df["plates_sold"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTrain size:", len(X_tr), "| Test size:", len(X_te))

# ---------- 3. TRAINING ----------
lr = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42).fit(X_tr, y_tr)

# ---------- 4. EVALUATION ----------
def ev(name, m):
    p = m.predict(X_te)
    cv = cross_val_score(m, X, y, cv=5, scoring="r2").mean()
    return dict(Model=name, MAE=mean_absolute_error(y_te, p),
                RMSE=mean_squared_error(y_te, p)**0.5, R2=r2_score(y_te, p), CV_R2=cv)
res = pd.DataFrame([ev("Linear Regression", lr), ev("Random Forest", rf)]).round(3)
print("\n=== MODEL COMPARISON (test set) ==="); print(res.to_string(index=False))
print("\n=== LINEAR REGRESSION COEFFICIENTS (standardised) ===")
print(pd.Series(lr[-1].coef_, index=X.columns).round(3))
print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")
print(pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).round(3))

# ---------- 5. RULE-BASED PREPARATION PLANNER ----------
def category(plates):
    return "High" if plates >= 300 else "Normal" if plates >= 180 else "Low"

def plan(attendance, temperature, rating, exam_week, weekend, model=rf):
    inp = pd.DataFrame([[attendance, temperature, rating, exam_week, weekend]], columns=X.columns)
    pred = float(max(model.predict(inp)[0], 0))
    cat = category(pred)
    # Predefined rules (NOT learned from data)
    buffer = 0.05 + (0.05 if exam_week else 0) + (0.05 if cat == "High" else 0)
    prepare = math.ceil(pred * (1 + buffer))
    slots = {"Breakfast": round(prepare*0.20), "Lunch": round(prepare*0.50), "Snacks": round(prepare*0.30)}
    return pred, cat, buffer, prepare, slots

print("\n=== PLANNER DEMO ===")
days = [("Day A", 85, 29, 4.2, 0, 0), ("Day B", 95, 33, 3.0, 1, 0), ("Day C", 50, 36, 3.5, 0, 1)]
for name, a, t, r, e, w in days:
    pr, c, b, prep, s = plan(a, t, r, e, w)
    print(f"\n{name}: attendance={a}%, temp={t}C, rating={r}, exam_week={e}, weekend={w}")
    print(f" Predicted plates: {pr:.1f} -> {c} | buffer {b:.0%} | prepare {prep}")
    for k, v in s.items(): print(f"    {k:<10}{v:>5} plates")

print("\n=== WHAT-IF: weekday, 30C, rating 3.5, varying attendance ===")
for a in (50, 65, 80, 95):
    pr, c, *_ = plan(a, 30, 3.5, 0, 0); print(f" {a:>3}% -> {pr:6.1f} ({c})")

# ---------- 6. PLANNER vs FIXED POLICY (test days) ----------
fixed = int(np.percentile(y_tr, 90))
prep_model = np.array([plan(*row.values)[3] for _, row in X_te.iterrows()])
actual = y_te.values
def sim(prep):
    return dict(prepared=int(prep.sum()), leftover=int(np.maximum(prep-actual, 0).sum()),
                shortage=int(np.maximum(actual-prep, 0).sum()), short_days=int((actual > prep).sum()))
print(f"\n=== FIXED POLICY ({fixed} plates/day) vs PLANNER, {len(actual)} test days, demand={actual.sum()} ===")
print("Fixed  :", sim(np.full(len(actual), fixed)))
print("Planner:", sim(prep_model))
