"""Shared leak-free data-preparation, ordinal meta-learning, and evaluation helpers for Lab 2."""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score

DATA_PATH = Path(__file__).parent / "data" / "AmesHousing.csv"

PRICE_BINS = [0, 100_000, 130_000, 180_000, 250_000, 400_000, np.inf]
LOG_BIN_EDGES = np.log1p([0.0, 100_000.0, 130_000.0, 180_000.0, 250_000.0, 400_000.0, np.inf])
CLASS_NAMES = ["<100k", "100-130k", "130-180k", "180-250k", "250-400k", ">=400k"]
N_CLASSES = len(CLASS_NAMES)

NOMINAL = [
    "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config",
    "Neighborhood", "Condition 1", "Condition 2", "Bldg Type", "House Style",
    "Roof Style", "Roof Matl", "Exterior 1st", "Exterior 2nd", "Mas Vnr Type",
    "Foundation", "Heating", "Central Air", "Electrical", "Garage Type",
    "Misc Feature", "Sale Type", "Sale Condition",
]

QUAL_SCALE = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
BSMT_FIN_SCALE = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}

ORDINAL_MAPS = {
    **{c: QUAL_SCALE for c in [
        "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC",
        "Kitchen Qual", "Fireplace Qu", "Garage Qual", "Garage Cond", "Pool QC",
    ]},
    "Bsmt Exposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFin Type 1": BSMT_FIN_SCALE,
    "BsmtFin Type 2": BSMT_FIN_SCALE,
    "Garage Finish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "Paved Drive": {"N": 0, "P": 1, "Y": 2},
    "Lot Shape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "Land Slope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}

NONE_CATEGORICAL = [
    "Alley", "Mas Vnr Type", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1",
    "BsmtFin Type 2", "Fireplace Qu", "Garage Type", "Garage Finish", "Garage Qual",
    "Garage Cond", "Pool QC", "Fence", "Misc Feature",
]

ZERO_NUMERIC = [
    "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
    "Bsmt Full Bath", "Bsmt Half Bath", "Garage Cars", "Garage Area",
]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def make_target(price: pd.Series) -> np.ndarray:
    """Map SalePrice to the six ordered classes 0..5 (left-closed intervals)."""
    return pd.cut(price, PRICE_BINS, right=False, labels=False).to_numpy(dtype=int)


def outlier_mask(df: pd.DataFrame) -> np.ndarray:
    """True for huge (>4000 sq ft) partial-sale outliers. Apply to training folds only."""
    return ((df["Gr Liv Area"] > 4000) & (df["Sale Condition"] == "Partial")).to_numpy(dtype=bool)


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


class AmesFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Leak-free preprocessing and domain feature engineering pipeline.
    Learns neighborhood medians, modes, and category schemas strictly on the training split in .fit().
    """
    def __init__(self, encoding_mode: str = "onehot", add_domain_features: bool = True):
        self.encoding_mode = encoding_mode
        self.add_domain_features = add_domain_features

    def _base_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.drop(columns=["Order", "PID", "SalePrice"], errors="ignore").copy()
        for col in NONE_CATEGORICAL:
            if col in X.columns:
                X[col] = X[col].fillna("None")
        for col in ZERO_NUMERIC:
            if col in X.columns:
                X[col] = X[col].fillna(0.0)

        if "Garage Yr Blt" in X.columns:
            X.loc[X["Garage Yr Blt"] > 2015, "Garage Yr Blt"] = 2007.0
            X["Garage Yr Blt"] = X["Garage Yr Blt"].fillna(X["Year Built"])

        if "Year Remod/Add" in X.columns and "Year Built" in X.columns:
            X["Year Remod/Add"] = np.maximum(X["Year Remod/Add"], X["Year Built"])

        X["MS SubClass"] = X["MS SubClass"].astype(str)
        return X

    def fit(self, X: pd.DataFrame, y=None):
        Xc = self._base_clean(X)
        self.elec_mode_ = Xc["Electrical"].mode(dropna=True).iloc[0]
        self.lot_frontage_nbhd_ = Xc.groupby("Neighborhood")["Lot Frontage"].median().to_dict()
        self.lot_frontage_global_ = float(Xc["Lot Frontage"].median())

        total_sf = Xc["Gr Liv Area"] + Xc["Total Bsmt SF"].fillna(0)
        Xc_temp = Xc.assign(_Total_SF=total_sf)
        self.nbhd_median_sf_ = Xc_temp.groupby("Neighborhood")["_Total_SF"].median().to_dict()
        self.nbhd_median_qual_ = Xc_temp.groupby("Neighborhood")["Overall Qual"].median().to_dict()
        self.nbhd_median_age_ = (Xc_temp["Yr Sold"] - Xc_temp["Year Built"]).groupby(Xc_temp["Neighborhood"]).median().to_dict()

        self.global_median_sf_ = float(total_sf.median())
        self.global_median_qual_ = float(Xc["Overall Qual"].median())
        self.global_median_age_ = float((Xc["Yr Sold"] - Xc["Year Built"]).median())

        X_trans = self._transform_internal(Xc, is_fit=True)
        self.feature_names_out_ = list(X_trans.columns)
        return self

    def _transform_internal(self, Xc: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        Xc = Xc.copy()
        Xc["Electrical"] = Xc["Electrical"].fillna(self.elec_mode_)
        nbhd_lf = Xc["Neighborhood"].map(self.lot_frontage_nbhd_)
        Xc["Lot Frontage"] = Xc["Lot Frontage"].fillna(nbhd_lf).fillna(self.lot_frontage_global_)

        num_cols = Xc.select_dtypes(include=[np.number]).columns
        Xc[num_cols] = Xc[num_cols].fillna(0.0)

        if self.encoding_mode == "label":
            for col in Xc.select_dtypes(exclude=[np.number]).columns:
                if is_fit:
                    if not hasattr(self, "label_maps_"):
                        self.label_maps_ = {}
                    cats = sorted(Xc[col].astype(str).unique())
                    self.label_maps_[col] = {k: i for i, k in enumerate(cats)}
                Xc[col] = Xc[col].astype(str).map(self.label_maps_.get(col, {})).fillna(-1).astype(int)
            return Xc

        for col, mapping in ORDINAL_MAPS.items():
            if col in Xc.columns:
                Xc[col] = Xc[col].map(mapping).fillna(0).astype(float)

        if self.add_domain_features:
            Xc["Total_Living_SF"] = Xc["Gr Liv Area"] + Xc["Total Bsmt SF"]
            Xc["Total_Finished_SF"] = Xc["Gr Liv Area"] + Xc["BsmtFin SF 1"] + Xc["BsmtFin SF 2"] - Xc["Low Qual Fin SF"]
            Xc["Finished_Area_Ratio"] = Xc["Total_Finished_SF"] / (Xc["Total_Living_SF"] + 1.0)
            Xc["Living_to_Lot_Ratio"] = Xc["Gr Liv Area"] / (Xc["Lot Area"] + 1.0)
            Xc["Log_Lot_Area"] = np.log1p(np.maximum(0, Xc["Lot Area"]))
            Xc["Log_Total_Living_SF"] = np.log1p(np.maximum(0, Xc["Total_Living_SF"]))

            Xc["Total_Baths"] = (
                Xc["Full Bath"] + 0.5 * Xc["Half Bath"] +
                Xc["Bsmt Full Bath"] + 0.5 * Xc["Bsmt Half Bath"]
            )
            Xc["Bath_to_Bed_Ratio"] = Xc["Total_Baths"] / (Xc["Bedroom AbvGr"] + 1.0)
            Xc["Avg_Room_Size"] = Xc["Gr Liv Area"] / (Xc["TotRms AbvGrd"] + 1.0)
            Xc["Porch_Deck_SF"] = (
                Xc["Wood Deck SF"] + Xc["Open Porch SF"] +
                Xc["Enclosed Porch"] + Xc["3Ssn Porch"] + Xc["Screen Porch"]
            )
            Xc["Bsmt_Fin_Share"] = (Xc["BsmtFin SF 1"] + Xc["BsmtFin SF 2"]) / (Xc["Total Bsmt SF"] + 1.0)

            Xc["Building_Age"] = np.maximum(0.0, Xc["Yr Sold"] - Xc["Year Built"])
            Xc["Remodel_Age"] = np.maximum(0.0, Xc["Yr Sold"] - Xc["Year Remod/Add"])
            Xc["Garage_Age"] = np.maximum(0.0, Xc["Yr Sold"] - Xc["Garage Yr Blt"])
            Xc["Is_New_Home"] = (Xc["Yr Sold"] == Xc["Year Built"]).astype(float)
            Xc["Is_Remodeled"] = (Xc["Year Remod/Add"] > Xc["Year Built"]).astype(float)
            Xc["Depreciation_Index"] = np.sqrt(Xc["Building_Age"]) + 0.5 * np.sqrt(Xc["Remodel_Age"])

            Xc["Overall_Qual_Sq"] = Xc["Overall Qual"] ** 2
            Xc["Luxury_Volume_Index"] = Xc["Overall_Qual_Sq"] * Xc["Log_Total_Living_SF"]
            Xc["Qual_x_Total_SF"] = Xc["Overall Qual"] * Xc["Total_Living_SF"]
            Xc["Qual_x_GrLiv"] = Xc["Overall Qual"] * Xc["Gr Liv Area"]
            Xc["Composite_Quality"] = (
                Xc["Overall Qual"] + Xc["Exter Qual"] + Xc["Kitchen Qual"] +
                Xc["Bsmt Qual"] + Xc["Garage Qual"] + Xc["Heating QC"] + Xc["Fireplace Qu"]
            )
            Xc["Qual_Cond_Interaction"] = Xc["Overall Qual"] * Xc["Overall Cond"]
            Xc["Garage_Composite"] = Xc["Garage Cars"] * Xc["Garage Area"] * (Xc["Garage Qual"] + 1.0)
            Xc["Has_2nd_Flr"] = (Xc["2nd Flr SF"] > 0).astype(float)
            Xc["Has_Pool"] = (Xc["Pool Area"] > 0).astype(float)

            nbhd_sf = Xc["Neighborhood"].map(self.nbhd_median_sf_).fillna(self.global_median_sf_)
            nbhd_qual = Xc["Neighborhood"].map(self.nbhd_median_qual_).fillna(self.global_median_qual_)
            nbhd_age = Xc["Neighborhood"].map(self.nbhd_median_age_).fillna(self.global_median_age_)

            Xc["Nbhd_Baseline_SF"] = nbhd_sf
            Xc["Nbhd_Baseline_Qual"] = nbhd_qual
            Xc["Rel_SF_to_Nbhd"] = Xc["Total_Living_SF"] / (nbhd_sf + 1.0)
            Xc["Qual_Diff_from_Nbhd"] = Xc["Overall Qual"] - nbhd_qual
            Xc["Age_Diff_from_Nbhd"] = Xc["Building_Age"] - nbhd_age

        if self.encoding_mode == "onehot":
            Xc = pd.get_dummies(Xc, columns=NOMINAL, dtype=float)
            Xc.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt") for c in Xc.columns]
            if not is_fit:
                Xc = Xc.reindex(columns=self.feature_names_out_, fill_value=0.0)
        elif self.encoding_mode == "native_cat":
            if is_fit:
                self.cat_categories_ = {col: Xc[col].astype("category").cat.categories for col in NOMINAL}
            for col in NOMINAL:
                Xc[col] = pd.Categorical(Xc[col], categories=self.cat_categories_[col])
        return Xc

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = self._base_clean(X)
        return self._transform_internal(Xc, is_fit=False)


class OrdinalPriceMetaGenerator:
    """Out-of-fold continuous log-price ensemble and Gaussian interval probability generator."""
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def _make_regressors(self):
        import xgboost as xgb
        import lightgbm as lgb
        xgb_reg = xgb.XGBRegressor(
            n_estimators=1100, learning_rate=0.025, max_depth=4,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=2.0,
            reg_lambda=2.0, reg_alpha=0.2, tree_method="hist",
            random_state=self.random_state, n_jobs=-1
        )
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.025, num_leaves=20,
            subsample_for_bin=50000, subsample=0.8, colsample_bytree=0.5,
            min_child_samples=15, reg_lambda=2.0, random_state=self.random_state,
            n_jobs=-1, verbose=-1
        )
        ridge_reg = Ridge(alpha=120.0)
        return xgb_reg, lgb_reg, ridge_reg

    def _append_meta_columns(self, X_df: pd.DataFrame, pred_xgb: np.ndarray,
                             pred_lgb: np.ndarray, pred_ridge: np.ndarray, sigma: float) -> pd.DataFrame:
        X_out = X_df.copy()
        blend = 0.50 * pred_xgb + 0.35 * pred_lgb + 0.15 * pred_ridge
        X_out["Meta_LogPrice_XGB"] = pred_xgb
        X_out["Meta_LogPrice_LGB"] = pred_lgb
        X_out["Meta_LogPrice_Ridge"] = pred_ridge
        X_out["Meta_LogPrice_Blend"] = blend
        X_out["Meta_Model_Spread"] = np.std(np.vstack([pred_xgb, pred_lgb, pred_ridge]), axis=0)

        for idx, thr in enumerate(LOG_BIN_EDGES[1:-1]):
            X_out[f"Meta_Dist_Thr_{idx}"] = blend - thr

        for c in range(N_CLASSES):
            z_low = (LOG_BIN_EDGES[c] - blend) / sigma
            z_high = (LOG_BIN_EDGES[c + 1] - blend) / sigma
            X_out[f"Meta_GaussProb_Class_{c}"] = stats.norm.cdf(z_high) - stats.norm.cdf(z_low)

        return X_out

    def fit_transform_oof(self, X_train: pd.DataFrame, log_y_train: np.ndarray) -> pd.DataFrame:
        n = len(X_train)
        oof_xgb, oof_lgb, oof_ridge = np.zeros(n), np.zeros(n), np.zeros(n)
        sigma_by_row = np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr_i, va_i in kf.split(X_train):
            xgb_r, lgb_r, ridge_r = self._make_regressors()
            scaler = StandardScaler()

            X_tr_f, X_va_f = X_train.iloc[tr_i], X_train.iloc[va_i]
            y_tr_f = log_y_train[tr_i]

            xgb_r.fit(X_tr_f, y_tr_f)
            lgb_r.fit(X_tr_f, y_tr_f)
            ridge_r.fit(scaler.fit_transform(X_tr_f), y_tr_f)

            train_blend = (
                0.50 * xgb_r.predict(X_tr_f) +
                0.35 * lgb_r.predict(X_tr_f) +
                0.15 * ridge_r.predict(scaler.transform(X_tr_f))
            )
            sigma_by_row[va_i] = max(float(np.std(y_tr_f - train_blend, ddof=1)), 1e-6)

            oof_xgb[va_i] = xgb_r.predict(X_va_f)
            oof_lgb[va_i] = lgb_r.predict(X_va_f)
            oof_ridge[va_i] = ridge_r.predict(scaler.transform(X_va_f))

        oof_blend = 0.50 * oof_xgb + 0.35 * oof_lgb + 0.15 * oof_ridge
        self.sigma_oof_ = float(np.mean(sigma_by_row))

        self.xgb_full_, self.lgb_full_, self.ridge_full_ = self._make_regressors()
        self.scaler_full_ = StandardScaler()
        self.xgb_full_.fit(X_train, log_y_train)
        self.lgb_full_.fit(X_train, log_y_train)
        self.ridge_full_.fit(self.scaler_full_.fit_transform(X_train), log_y_train)

        return self._append_meta_columns(X_train, oof_xgb, oof_lgb, oof_ridge, sigma_by_row)

    def transform_test(self, X_test: pd.DataFrame) -> pd.DataFrame:
        p_xgb = self.xgb_full_.predict(X_test)
        p_lgb = self.lgb_full_.predict(X_test)
        p_ridge = self.ridge_full_.predict(self.scaler_full_.transform(X_test))
        return self._append_meta_columns(X_test, p_xgb, p_lgb, p_ridge, self.sigma_oof_)


def predict_with_weights(proba: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.argmax(proba * np.asarray(weights)[None, :], axis=1)


def optimize_class_weights(proba: np.ndarray, y: np.ndarray, n_rounds: int = 4, grid=np.linspace(0.45, 2.50, 83)):
    w = np.ones(proba.shape[1], dtype=float)
    best = macro_f1(y, predict_with_weights(proba, w))
    for _ in range(n_rounds):
        improved = False
        for k in range(len(w)):
            for g in grid:
                cand = w.copy()
                cand[k] = g
                score = macro_f1(y, predict_with_weights(proba, cand))
                if score > best + 1e-6:
                    best, w, improved = score, cand, True
        if not improved:
            break
    return w, best