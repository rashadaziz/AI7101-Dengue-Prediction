# %% [markdown]
# # DengAI: Predicting Weekly Dengue Cases
#
# Predict weekly dengue cases in San Juan (`sj`) and Iquitos (`iq`) from weather data, so health teams can prepare early.
# Metric: **MAE** (average number of cases we are off per week). It is easy to explain to health teams, and less dominated by the rare outbreak weeks (1.6) than RMSE.

# %%
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sns.set()

# %%
train = pd.read_csv("data/dengue_features_train.csv", parse_dates=["week_start_date"]).merge(
        pd.read_csv("data/dengue_labels_train.csv"), on=["city", "year", "weekofyear"])

test = pd.read_csv("data/dengue_features_test.csv", parse_dates=["week_start_date"])

WEATHER = [c for c in train.columns if c.startswith(("ndvi", "precip", "reanalysis", "station"))]

# %% [markdown]
# ## 1. Exploratory Data Analysis
#
# ### 1.1 Cases over time

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 7))

for i, city in enumerate(["sj", "iq"]):
    df = train[train.city == city]
    sns.lineplot(data=df, x="week_start_date", y="total_cases", ax=axes[i, 0]).set_title(f"{city}: cases over time")
    sns.lineplot(data=df, x="weekofyear", y="total_cases", estimator="median", ax=axes[i, 1]).set_title(f"{city}: median cases by week")

plt.tight_layout()
plt.show()

# %% [markdown]
# Cases are seasonal with a few big outbreaks, and the cities are very different (San Juan has far more cases and a different peak week), so we train **one model per city** and give the model the week of the year as `sin`/`cos` features.

# %% [markdown]
# ### 1.2 Missing values

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

for ax, (city, df) in zip(axes, train.groupby("city")):
    sns.heatmap(df[WEATHER].isna().T.astype(int), cbar=False, yticklabels=True, ax=ax)
    ax.set_title(f"{city}: missing values (light = missing)"); ax.set_xticks([])

plt.tight_layout()
plt.show()

print("Max difference between the two precipitation columns:",
      (train["precipitation_amt_mm"] - train["reanalysis_sat_precip_amt_mm"]).abs().max())

# %% [markdown]
# Gaps are short and scattered (the longest is 15 weeks, mostly `ndvi_ne` in San Juan), and weather changes slowly, so we **fill each gap with the previous week's value**.
# `reanalysis_sat_precip_amt_mm` is an exact copy of `precipitation_amt_mm`, so we **drop it**.

# %% [markdown]
# ### 1.3 Correlation with cases
# The **bold** features are the ones we average over past weeks in preprocessing (`KEY`).

# %%
KEY = [
    "station_avg_temp_c",
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
    "reanalysis_min_air_temp_k",
    "precipitation_amt_mm"
]

corr = pd.DataFrame({city: df[WEATHER + ["total_cases"]].corr()["total_cases"].drop("total_cases")
                     for city, df in train.groupby("city")})

corr = corr.loc[corr.abs().mean(axis=1).sort_values().index]
ax = corr.plot.barh(figsize=(8, 8), title="Correlation with total_cases (same week)")
for label in ax.get_yticklabels():
    label.set_fontweight("bold" if label.get_text() in KEY else "normal")

plt.show()

# %% [markdown]
# Humidity, dew point and minimum temperature are the top three features in both cities, and average temperature is among the strongest in San Juan, so those are in `KEY`.
# Precipitation is weak in the same week, but it is how mosquitoes get standing water to breed in, so its effect should show up weeks later.

# %% [markdown]
# ### 1.4 Why windowed features
# Correlation of cases with the average of each `KEY` feature over the last *w* weeks (*w* = 1 is the raw weekly value).

# %%
windows = range(1, 17)
fig, axes = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
for ax, (city, df) in zip(axes, train.groupby("city")):
    for c in KEY:
        x = df[c].ffill()
        ax.plot(windows, [df["total_cases"].corr(x.rolling(w, min_periods=1).mean()) for w in windows], marker="o", label=c)
    for w in [4, 8, 12]:
        ax.axvline(w, color="grey", ls=":")
    ax.set(title=f"{city}: correlation of the w-week average with cases", xlabel="window w (weeks)")
axes[0].set_ylabel("correlation with total_cases"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# Averaging over past weeks makes every `KEY` feature more predictive: in San Juan the correlation keeps rising up to 12–16 weeks (average temperature goes from 0.19 to 0.37 at 12 weeks), in Iquitos it peaks at 4–8 weeks, and precipitation goes from 0.06 to 0.16 in San Juan and from 0.09 to 0.16 in Iquitos.
# This fits the biology: mosquitoes take weeks to breed and the virus takes time to incubate, so the last few weeks of weather matter more than this week's.
# **4, 8 and 12-week averages** cover the best windows of both cities.

# %% [markdown]
# ### 1.5 Cases per season

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

for ax, (city, df) in zip(axes, train.groupby("city")):
    season = np.arange(len(df)) // 52
    med = df["total_cases"].groupby(season.tolist()).median()
    med.index = df["week_start_date"].iloc[::52].dt.year.values[:len(med)]
    med.plot.bar(ax=ax, title=f"{city}: median weekly cases per season")

plt.tight_layout(); plt.show()

# %% [markdown]
# The level changes a lot from season to season: San Juan's 1990s seasons run 2–3 times higher than its 2000s, and Iquitos reports almost no cases in its first seasons.
# Weeks inside a season move together, so we **validate on whole seasons, always later than the training data** (section 3), and Iquitos gets at least 5 training seasons before its first fold.

# %% [markdown]
# ### 1.6 Target distribution

# %%
sns.displot(train, x="total_cases", col="city", bins=50, height=4, aspect=1.6,
            facet_kws={"sharex": False, "sharey": False})
plt.show()

print(train.groupby("city")["total_cases"].describe().round(1))

# %% [markdown]
# Weekly cases are strongly right-skewed in both cities: most weeks are low (median 19 in San Juan, 5 in Iquitos), but outbreak weeks reach 461 and 116, and 18% of Iquitos weeks have zero cases.
# We **keep the raw counts** instead of a log transform: the competition scores MAE on raw counts, and XGBoost with `reg:absoluteerror` optimises that directly, predicting the median, which is not pulled up by the rare outbreak weeks.

# %% [markdown]
# ## 2. Preprocessing
# - `city` is the only categorical variable: instead of one-hot encoding it, we train **one model per city**, because the cities differ in scale and peak week (1.1).
# - Fill missing values with the previous week's value (1.2).
# - Drop the duplicate precipitation column (1.2).
# - Add season features (`sin`/`cos` of the week) (1.1): week of year is cyclical, week 52 is next to week 1. Encoding it with sine and cosine puts the weeks on a circle, so the model sees the seasons as continuous, which matters because Iquitos's dengue season crosses the new year.
# - Add the average of the `KEY` features over the last 4, 8 and 12 weeks (1.3, 1.4).
# - Split into inputs `X` (all weather and season features) and output `y` (`total_cases`).
# - Keep `total_cases` as raw counts, no transform (1.6).

# %%
train["part"], test["part"] = "train", "test"
data = pd.concat([train, test]).drop(columns="reanalysis_sat_precip_amt_mm")
data = data.sort_values(["city", "week_start_date"]).reset_index(drop=True)

WEATHER.remove("reanalysis_sat_precip_amt_mm")
data[WEATHER] = data.groupby("city")[WEATHER].ffill()

data["woy_sin"] = np.sin(2 * np.pi * data["weekofyear"] / 52)
data["woy_cos"] = np.cos(2 * np.pi * data["weekofyear"] / 52)

for w in [4, 8, 12]:
    for c in KEY:
        data[f"{c}_{w}w"] = data.groupby("city")[c].transform(lambda s: s.rolling(w, min_periods=1).mean())

FEATURES = [c for c in data.columns if c not in ["city", "year", "weekofyear", "week_start_date", "total_cases", "part"]]
X = data[FEATURES]          # inputs
y = data["total_cases"]     # output (NaN for test rows)
print(len(FEATURES), "features")

# %% [markdown]
# ## 3. Validation
# - **Outer loop (scoring):** for each season *s*, train on all earlier seasons and predict *s*. San Juan starts at season 8 (10 folds), Iquitos at season 5 (5 folds; its first years have almost no cases).
# - **Inner loop (tuning):** `GridSearchCV` inside each outer training set, validating on its last 2 seasons the same way.

# %%
MIN_TRAIN = {"sj": 8, "iq": 5}

def season_folds(n_rows, first):
    # Expanding window: for each season s >= first, train on all seasons before s, validate on s
    season = np.arange(n_rows) // 52
    for s in range(first, season.max() + 1):
        yield np.where(season < s)[0], np.where(season == s)[0]

# %% [markdown]
# ## 4. Model choosing
# Three models plus a baseline, each scored on every outer season (the models are tuned in the inner loop):
# - **Ridge** (simple linear model)
# - **Random Forest**
# - **XGBoost** (gradient boosting, optimising MAE directly)
# - **Baseline**: median cases of the same week of the year in the training seasons (no weather). A model is only useful if it beats this.

# %%
MODELS = {
    "Ridge": (make_pipeline(StandardScaler(), Ridge()),
              {"ridge__alpha": [1, 10, 100, 1000, 10000]}),
    "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=42),
                      {"max_depth": [3, 5, None], "min_samples_leaf": [1, 5, 20]}),
    "XGBoost": (XGBRegressor(objective="reg:absoluteerror", learning_rate=0.05, random_state=42),
                {"max_depth": [2, 3, 4], "n_estimators": [100, 300]}),
}

rows, preds = [], []
for city in ["sj", "iq"]:
    d = data[(data.city == city) & (data.part == "train")]
    for train_idx, val_idx in season_folds(len(d), MIN_TRAIN[city]):
        tr, va = d.iloc[train_idx], d.iloc[val_idx]
        X_tr, y_tr = X.loc[tr.index], y.loc[tr.index]
        X_va, y_va = X.loc[va.index], y.loc[va.index]
        inner = list(season_folds(len(tr), len(tr) // 52 - 2))
        # Baseline: median cases of the same week in past seasons, no weather
        wk_median = tr.groupby("weekofyear")["total_cases"].median()
        base = va["weekofyear"].map(wk_median).fillna(y_tr.median())
        rows.append({"city": city, "season": va["week_start_date"].iloc[0].year, "model": "Baseline",
                     "MAE": mean_absolute_error(y_va, base)})
        preds.append(va[["city", "week_start_date", "total_cases"]].assign(model="Baseline", pred=base))
        for name, (model, grid) in MODELS.items():
            gs = GridSearchCV(model, grid, cv=inner, scoring="neg_mean_absolute_error")
            gs.fit(X_tr, y_tr)
            pred = gs.predict(X_va).clip(0)
            rows.append({"city": city, "season": va["week_start_date"].iloc[0].year, "model": name,
                        "MAE": mean_absolute_error(y_va, pred)})
            preds.append(va[["city", "week_start_date", "total_cases"]].assign(model=name, pred=pred))

scores = pd.DataFrame(rows).pivot_table(index=["city", "season"], columns="model", values="MAE")
scores.round(1)

# %% [markdown]
# ## 5. Results
# One MAE per model per season. Outbreak seasons dominate the mean, so also look at the standard error and at how many seasons each model wins.

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
for ax, city in zip(axes, ["sj", "iq"]):
    scores.loc[city].plot.bar(ax=ax, title=f"{city}: MAE per validation season")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Model choice table.** `gap vs best` is how many more cases per week a model is off than the best model, averaged over the same seasons (± its std err).
# A gap smaller than its std err is noise, so that model is a **tie** with the best one.

# %%
table = []
for city, s in scores.groupby("city"):
    best = s.mean().idxmin()
    gap = s.sub(s[best], axis=0)              # extra error vs the best model, season by season
    wins = s.idxmin(axis=1).value_counts()
    for m in s.columns:
        table.append({"city": city, "model": m, "mean MAE": s[m].mean(), "std err": s[m].sem(),
                      "seasons won": f"{wins.get(m, 0)}/{len(s)}",
                      "gap vs best": f"{gap[m].mean():+.1f} ± {gap[m].sem():.1f}",
                      "verdict": "best" if m == best else "tie" if gap[m].mean() <= gap[m].sem() else "worse"})
choice = pd.DataFrame(table).set_index(["city", "model"]).sort_values(["city", "mean MAE"])
choice.round(2)

# %% [markdown]
# XGBoost is the best ML model in both cities, but it only ties with the seasonal baseline (sj 20.1 vs 19.8, iq 6.0 vs 6.2): most of the predictable signal is seasonality, and weather helps mainly in some outbreak seasons (e.g. sj 2005: 17.9 vs 23.5). We keep XGBoost as the final model.

# %%
FINAL = {"sj": "XGBoost", "iq": "XGBoost"}  # set from the table above

# %% [markdown]
# **Predicted vs actual.** Out-of-fold predictions of the final model and the baseline on every validation season.

# %%
p = pd.concat(preds)
fig, axes = plt.subplots(2, 1, figsize=(15, 8))
for ax, city in zip(axes, ["sj", "iq"]):
    d = p[p.city == city]
    sns.lineplot(data=d[d.model == "Baseline"], x="week_start_date", y="total_cases", label="actual", color="black", ax=ax)
    for m in [FINAL[city], "Baseline"]:
        sns.lineplot(data=d[d.model == m], x="week_start_date", y="pred", label=m, ax=ax)
    ax.set_title(f"{city}: actual vs predicted cases (validation seasons)")
plt.tight_layout(); plt.show()

# %% [markdown]
# In normal seasons XGBoost follows the timing of the cases, but it never predicts outbreaks: its highest San Juan prediction is about 70 cases, while the real peaks were 329 (1998), 137 (2005) and 170 (2007), and in Iquitos it predicts about 10 when the peaks were 58 and 63.
# The baseline has the same ceiling, which is why the two tie: weather tells the model *when* the season comes, not *how big* it will be.

# %% [markdown]
# ## 6. Training (final model)
# Tune the chosen model on all training seasons with the same season folds, refit on everything and predict the competition test set.

# %%
parts = []
city_model = {}
for city in ["sj", "iq"]:
    d_train = data[(data.city == city) & (data.part == "train")]
    d_test = data[(data.city == city) & (data.part == "test")]

    model, grid = MODELS[FINAL[city]]
    gs = GridSearchCV(model, grid, cv=list(season_folds(len(d_train), MIN_TRAIN[city])), scoring="neg_mean_absolute_error")
    gs.fit(X.loc[d_train.index], y.loc[d_train.index])
    print(city, gs.best_params_)

    city_model[city] = gs.best_estimator_
    parts.append(d_test[["city", "year", "weekofyear"]].assign(
        total_cases=np.round(gs.predict(X.loc[d_test.index]).clip(0)).astype(int)))

submission = pd.concat(parts)
submission.to_csv("submission.csv", index=False)
submission.groupby("city").total_cases.describe()

# %% [markdown]
# Both cities pick the simplest XGBoost in the grid (`max_depth=2`, 100 trees; CV MAE 20.0 in San Juan, 5.9 in Iquitos): with little signal beyond seasonality, bigger trees only fit noise.

# %% [markdown]
# ## 7. Post analysis

# %%
cities = ["sj", "iq"]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, city in zip(axes, cities):
    estimator = city_model[city]
    importances = estimator.feature_importances_

    imp_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances})

    imp_df = imp_df.reindex(
        imp_df.Importance.abs().sort_values(ascending=False).index
    )

    sns.barplot(
        data=imp_df,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title(f"{city.upper()} Best Model Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

plt.tight_layout()
plt.show()

# %% [markdown]
# The feature importance plot above shows that our engineered features (averaging features over past-weeks) contribute significantly to the model's prediction.
# On the test leaderbord, the model achieved a MAE of 24.6, meaning predictions deviate from the true case count by roughly 24 cases per week on average.
# As observed in Results, prediction errors widen considerably during peak outbreak seasons where extreme case spikes occur.
# Consequently, the model should not be used to micromanage medical supply quotas or justify reducing baseline resource allocations.
# Instead, it is best deployed as an early-warning system to ensure hospital surge readiness and improve public awareness.