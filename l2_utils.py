"""Shared data-preparation and evaluation helpers for the L2 classification lab.

Everything here is target-free (no SalePrice information leaks into features), so the
functions can be applied to the full dataset before splitting. The only target-dependent
step (dropping the partial-sale outliers) is exposed as a mask that must be applied to
training folds only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DATA_PATH = Path(__file__).parent / "data" / "AmesHousing.csv"

PRICE_BINS = [0, 100_000, 130_000, 180_000, 250_000, 400_000, np.inf]
CLASS_NAMES = ["<100k", "100-130k", "130-180k", "180-250k", "250-400k", ">=400k"]

NOMINAL = [
    "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config",
    "Neighborhood", "Condition 1", "Condition 2", "Bldg Type", "House Style",
    "Roof Style", "Roof Matl", "Exterior 1st", "Exterior 2nd", "Mas Vnr Type",
    "Foundation", "Heating", "Central Air", "Electrical", "Garage Type",
    "Misc Feature", "Sale Type", "Sale Condition",
]

# Ordinal scales from the data dictionary (AmesExplanations.txt); "None" = feature absent.
QUAL_SCALE = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPS = {
    **{c: QUAL_SCALE for c in [
        "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC",
        "Kitchen Qual", "Fireplace Qu", "Garage Qual", "Garage Cond", "Pool QC",
    ]},
    "Bsmt Exposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFin Type 1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFin Type 2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "Garage Finish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "Paved Drive": {"N": 0, "P": 1, "Y": 2},
    "Lot Shape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "Land Slope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}

# Missing values that mean "this house has no such feature" rather than "unknown".
NONE_CATEGORICAL = [
    "Alley", "Mas Vnr Type", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1",
    "BsmtFin Type 2", "Fireplace Qu", "Garage Type", "Garage Finish", "Garage Qual",
    "Garage Cond", "Pool QC", "Fence", "Misc Feature",
]
ZERO_NUMERIC = [
    "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
    "Bsmt Full Bath", "Bsmt Half Bath", "Garage Cars", "Garage Area",
]


def load_data(path=DATA_PATH):
    return pd.read_csv(path, sep="\t")


def make_target(price):
    """Map SalePrice to the six lab classes 0..5 (left-closed intervals)."""
    return pd.cut(price, PRICE_BINS, right=False, labels=False).astype(int)


def clean(df):
    """Fix missing values and data-entry errors. Returns a new frame without SalePrice/IDs."""
    df = df.drop(columns=["Order", "PID", "SalePrice"], errors="ignore").copy()

    df[NONE_CATEGORICAL] = df[NONE_CATEGORICAL].fillna("None")
    df[ZERO_NUMERIC] = df[ZERO_NUMERIC].fillna(0)
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    # Lot frontage is similar within a neighborhood; global median as a fallback.
    df["Lot Frontage"] = df.groupby("Neighborhood")["Lot Frontage"].transform(
        lambda s: s.fillna(s.median())
    )
    df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())

    # Garage Yr Blt: 2207 is a typo for 2007; houses without garage get the build year.
    df.loc[df["Garage Yr Blt"] > 2010, "Garage Yr Blt"] = 2007
    df["Garage Yr Blt"] = df["Garage Yr Blt"].fillna(df["Year Built"])

    # Remodel year cannot precede the build year.
    df["Year Remod/Add"] = df[["Year Remod/Add", "Year Built"]].max(axis=1)

    df["MS SubClass"] = df["MS SubClass"].astype(str)
    return df


def add_features(df):
    """Domain features: total size, age, bathrooms and quality x size interactions."""
    df = df.copy()
    df["Total SF"] = df["Gr Liv Area"] + df["Total Bsmt SF"]
    df["Finished SF"] = df["Gr Liv Area"] + df["BsmtFin SF 1"] + df["BsmtFin SF 2"]
    df["House Age"] = df["Yr Sold"] - df["Year Built"]
    df["Years Since Remodel"] = df["Yr Sold"] - df["Year Remod/Add"]
    df["Remodeled"] = (df["Year Remod/Add"] != df["Year Built"]).astype(int)
    df["Total Baths"] = (
        df["Full Bath"] + 0.5 * df["Half Bath"]
        + df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"]
    )
    df["Total Porch SF"] = df[
        ["Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Wood Deck SF"]
    ].sum(axis=1)
    df["Qual x GrLiv"] = df["Overall Qual"] * df["Gr Liv Area"]
    df["Qual x TotalSF"] = df["Overall Qual"] * df["Total SF"]
    df["Qual Sum"] = df["Overall Qual"] + df["Overall Cond"]
    df["Lot Area log"] = np.log1p(df["Lot Area"])
    df["Has 2nd Floor"] = (df["2nd Flr SF"] > 0).astype(int)
    df["Has Garage"] = (df["Garage Area"] > 0).astype(int)
    df["Has Fireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["Garage Score"] = df["Garage Cars"] * df["Garage Qual"]
    df["Kitchen Score"] = df["Kitchen AbvGr"] * df["Kitchen Qual"]
    df["Exter Score"] = df["Exter Qual"] * df["Exter Cond"]
    df["Bsmt Score"] = df["Bsmt Qual"] * df["Total Bsmt SF"]
    return df


def encode_ordinals(df):
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        df[col] = df[col].map(mapping).astype(float)
    return df


def prepare(df, strategy="onehot", features=True):
    """Full target-free pipeline.

    strategy:
      "onehot" - ordinal scales -> integers, nominal -> one-hot dummies
      "native" - ordinal scales -> integers, nominal -> pandas category (XGBoost/LightGBM native)
      "label"  - everything (incl. ordinals) -> arbitrary integer codes (naive baseline)
    """
    X = clean(df)
    if strategy == "label":
        for col in X.select_dtypes(exclude="number").columns:
            X[col] = X[col].astype("category").cat.codes
        return X
    X = encode_ordinals(X)
    if features:
        X = add_features(X)
    if strategy == "onehot":
        X = pd.get_dummies(X, columns=NOMINAL, dtype=np.uint8)
    elif strategy == "native":
        X[NOMINAL] = X[NOMINAL].astype("category")
    else:
        raise ValueError(strategy)
    X.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt") for c in X.columns]
    return X


def outlier_mask(df):
    """True for the huge (>4000 sq ft) partial sales that the data documentation
    recommends removing. Apply to training data only."""
    return ((df["Gr Liv Area"] > 4000) & (df["Sale Condition"] == "Partial")).to_numpy()


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def predict_with_weights(proba, weights):
    """argmax of class-rescaled probabilities: weights shift the decision thresholds."""
    return np.argmax(proba * np.asarray(weights)[None, :], axis=1)


def optimize_class_weights(proba, y, n_rounds=4, grid=np.linspace(0.5, 3.0, 51)):
    """Coordinate ascent on per-class probability multipliers to maximise macro-F1.

    Multiplying class k's probability by w_k is equivalent to moving the one-vs-rest
    decision threshold of class k, so this searches the precision/recall trade-off of
    every class jointly. Fit on out-of-fold probabilities, never on the test set.
    """
    w = np.ones(proba.shape[1])
    best = macro_f1(y, predict_with_weights(proba, w))
    for _ in range(n_rounds):
        improved = False
        for k in range(len(w)):
            for g in grid:
                cand = w.copy()
                cand[k] = g
                score = macro_f1(y, predict_with_weights(proba, cand))
                if score > best + 1e-9:
                    best, w, improved = score, cand, True
        if not improved:
            break
    return w, best


REG_PARAMS = dict(
    n_estimators=1500, learning_rate=0.02, max_depth=4, subsample=0.8,
    colsample_bytree=0.5, min_child_weight=2, n_jobs=-1, random_state=0,
)


def add_price_estimate(X_train, log_price_train, X_test=None, k=5, seed=1, params=REG_PARAMS):
    """Stacking feature: an XGBoost regression estimate of log(SalePrice).

    The class labels are ordered price bins, which a plain multiclass objective ignores.
    A regressor learns that order; its prediction is handed to the classifier as a feature.
    Training rows get *out-of-fold* predictions (each row predicted by a model that never
    saw it) so the classifier does not learn from leaked, over-optimistic values; test rows
    get the prediction of a regressor fitted on the whole training set.
    """
    import xgboost as xgb
    from sklearn.model_selection import KFold

    log_price_train = np.asarray(log_price_train)
    oof = np.zeros(len(X_train))
    for tr, va in KFold(k, shuffle=True, random_state=seed).split(X_train):
        model = xgb.XGBRegressor(**params).fit(X_train.iloc[tr], log_price_train[tr])
        oof[va] = model.predict(X_train.iloc[va])
    X_train = X_train.assign(**{"Price Estimate": oof})
    if X_test is None:
        return X_train
    full = xgb.XGBRegressor(**params).fit(X_train.drop(columns="Price Estimate"), log_price_train)
    return X_train, X_test.assign(**{"Price Estimate": full.predict(X_test)})
