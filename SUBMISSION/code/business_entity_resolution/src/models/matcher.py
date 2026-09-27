"""
LightGBM + XGBoost binary classifier ensemble for entity matching.

Trains on candidate pair features + ground truth labels.
Uses scale_pos_weight for class imbalance.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LGBM_PARAMS, OUTPUT_DIR, PIPELINE_VERSION, SAMPLE_FRAC
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'cache')

SAMPLE_KEY = f"{SAMPLE_FRAC:g}"

class EntityMatcher:
    """Ensemble matcher."""

    def __init__(self, params: dict | None = None):
        self.params = dict(LGBM_PARAMS)
        if params:
            self.params.update(params)
        self.lgb_model = None
        self.xgb_model = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Train both classifiers.
        """
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        self.params['scale_pos_weight'] = scale_pos_weight
        
        print(f"[matcher] Training Ensemble: {len(X_train):,} samples, scale_pos_weight={scale_pos_weight:.2f}")

        # --- LightGBM ---
        print("\n[matcher] Training LightGBM...")
        import joblib
        lgb_cache_path = os.path.join(CACHE_DIR, f'lgb_model_{PIPELINE_VERSION}_{SAMPLE_KEY}.pkl')
        if os.path.exists(lgb_cache_path):
            print("[matcher] Loading LightGBM from per-model cache...")
            self.lgb_model = joblib.load(lgb_cache_path)
        else:
            self.lgb_model = lgb.LGBMClassifier(**self.params)
            lgb_callbacks = [lgb.log_evaluation(period=50)]
            
            fit_params = {}
            if feature_names:
                fit_params['feature_name'] = feature_names
    
            if X_val is not None and y_val is not None:
                lgb_callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=True))
                self.lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='binary_logloss',
                    callbacks=lgb_callbacks,
                    **fit_params,
                )
            else:
                self.lgb_model.fit(X_train, y_train, callbacks=lgb_callbacks, **fit_params)
            
            joblib.dump(self.lgb_model, lgb_cache_path)

        # --- XGBoost ---
        print("\n[matcher] Training XGBoost...")
        xgb_cache_path = os.path.join(CACHE_DIR, f'xgb_model_{PIPELINE_VERSION}_{SAMPLE_KEY}.pkl')
        if os.path.exists(xgb_cache_path):
            print("[matcher] Loading XGBoost from per-model cache...")
            self.xgb_model = joblib.load(xgb_cache_path)
        else:
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'learning_rate': 0.05,
                'max_depth': 6,
                'n_estimators': 500,
                'scale_pos_weight': scale_pos_weight,
                'random_state': self.params.get('random_state', 42),
                'n_jobs': -1,
                'tree_method': 'hist', # fast histogram
                'early_stopping_rounds': 30 if X_val is not None else None,
            }
            self.xgb_model = xgb.XGBClassifier(**xgb_params)
            
            if X_val is not None and y_val is not None:
                self.xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=50,
                )
            else:
                self.xgb_model.fit(X_train, y_train, verbose=50)
            
            joblib.dump(self.xgb_model, xgb_cache_path)

        print("[matcher] Ensemble Training complete.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class (match) averaged."""
        assert self.lgb_model is not None and self.xgb_model is not None, "Models not trained"
        p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        return (p_lgb + p_xgb) / 2.0

    def feature_importance(self, feature_names: list[str] | None = None) -> dict[str, float]:
        """Return feature importance dict (from LightGBM)."""
        assert self.lgb_model is not None, "Model not trained"
        importances = self.lgb_model.feature_importances_
        if feature_names is not None:
            return dict(zip(feature_names, importances))
        return dict(enumerate(importances))

    def save(self, path: str | os.PathLike | None = None) -> None:
        if path is None:
            path = OUTPUT_DIR / "model.pkl"
        with open(path, 'wb') as f:
            pickle.dump({'lgb': self.lgb_model, 'xgb': self.xgb_model}, f)
        print(f"[matcher] Ensemble saved to {path}")

    def load(self, path: str | os.PathLike | None = None) -> None:
        import joblib
        lgb_path = os.path.join(CACHE_DIR, 'lgb_model.pkl')
        xgb_path = os.path.join(CACHE_DIR, 'xgb_model.pkl')
        if os.path.exists(lgb_path):
            self.lgb_model = joblib.load(lgb_path)
        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
        print(f"[matcher] Ensemble loaded from {CACHE_DIR}")
