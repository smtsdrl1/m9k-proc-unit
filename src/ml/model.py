"""
ML Signal Predictor — Self-learning model that trains on historical signal outcomes.

Features extracted from each signal:
- Technical: RSI, MACD histogram, ADX, Bollinger %B, Stochastic K, ATR%
- Volume: Volume ratio vs 20-period SMA
- Multi-timeframe: MTF alignment score
- Sentiment: Fear & Greed, social sentiment
- Smart Money: Institutional flow score
- Signal: Confidence score, tier level

Target: Binary WIN (any target hit) / LOSS (SL hit)

Auto-retrains when ≥20 new outcomes accumulate since last training.
"""
import io
import json
import logging
import pickle
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger("matrix_trader.ml")

# Feature columns used for training
FEATURE_NAMES = [
    "rsi", "macd_hist", "adx", "bb_pctb", "stoch_k", "atr_pct",
    "volume_ratio", "mtf_score", "sentiment_score", "fear_greed",
    "smart_money_score", "macro_score", "confidence", "tier_numeric",
    "is_crypto",
]

MIN_TRAINING_SAMPLES = 20   # Minimum signals to train on
RETRAIN_THRESHOLD = 15      # Retrain after this many new outcomes


class SignalPredictor:
    """ML model that learns from signal outcomes to predict success probability."""

    def __init__(self, db=None):
        from src.database.db import Database
        self.db = db or Database()
        self.model = None
        self.scaler = None
        self.feature_names = FEATURE_NAMES
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        """Load the latest trained model from database."""
        try:
            model_data = self.db.get_latest_ml_model("signal_predictor")
            if model_data and model_data.get("model_data"):
                state = pickle.loads(model_data["model_data"])
                self.model = state.get("model")
                self.scaler = state.get("scaler")
                self.feature_names = model_data.get("feature_names", FEATURE_NAMES)
                self.is_loaded = True
                logger.info(
                    f"ML model loaded — accuracy: {model_data['accuracy']:.1f}%, "
                    f"samples: {model_data['total_samples']}"
                )
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")
            self.is_loaded = False

    def extract_features(self, indicators: dict, mtf_result: dict = None,
                         sentiment: dict = None, smart_money: dict = None,
                         macro: dict = None, confidence: int = 0,
                         tier: str = "", is_crypto: bool = True) -> dict:
        """Extract feature vector from analysis components.
        All values come from live market data — no random/fake data.
        """
        # Technical indicators (from real OHLCV data)
        rsi = indicators.get("rsi", 50)
        macd_hist = indicators.get("macd_histogram", 0)
        adx = indicators.get("adx", 20)
        bb_pctb = indicators.get("bb_percent_b", 0.5)
        stoch_k = indicators.get("stochastic_k", 50)
        atr = indicators.get("atr", 0)
        price = indicators.get("close", indicators.get("price", 0))
        atr_pct = (atr / price * 100) if price > 0 else 0

        # Volume (from real OHLCV)
        volume_ratio = indicators.get("volume_ratio", 1.0)

        # Multi-timeframe (calculated from real data across timeframes)
        mtf_result = mtf_result or {}
        mtf_score = mtf_result.get("alignment_score", 50)

        # Sentiment (from CoinGlass / alternative.me APIs)
        sentiment = sentiment or {}
        sentiment_score = sentiment.get("score", 50)
        fear_greed = sentiment.get("fear_greed", 50)

        # Smart Money (from real market depth / funding rate data)
        smart_money = smart_money or {}
        smart_money_score = smart_money.get("score", 50)

        # Macro (from real economic calendar / DXY data)
        macro = macro or {}
        macro_score = macro.get("score", 50)

        # Tier to numeric
        tier_map = {"SNIPER_1": 6, "SNIPER_2": 5, "SNIPER_3": 4,
                    "SNIPER_4": 3, "SNIPER_5": 2, "SNIPER_6": 1}
        tier_numeric = tier_map.get(tier, 0)

        return {
            "rsi": round(rsi, 2),
            "macd_hist": round(macd_hist, 6),
            "adx": round(adx, 2),
            "bb_pctb": round(bb_pctb, 4),
            "stoch_k": round(stoch_k, 2),
            "atr_pct": round(atr_pct, 4),
            "volume_ratio": round(volume_ratio, 4),
            "mtf_score": round(mtf_score, 2),
            "sentiment_score": round(sentiment_score, 2),
            "fear_greed": round(fear_greed, 2),
            "smart_money_score": round(smart_money_score, 2),
            "macro_score": round(macro_score, 2),
            "confidence": confidence,
            "tier_numeric": tier_numeric,
            "is_crypto": int(is_crypto),
        }

    def predict(self, features: dict) -> Optional[dict]:
        """Predict signal outcome probability.
        Returns: {"win_probability": 0.0-1.0, "confidence_adjustment": int}
        or None if model not trained yet.
        """
        if not self.is_loaded or self.model is None:
            return None

        try:
            # Build feature vector in correct order
            X = np.array([[features.get(f, 0) for f in self.feature_names]])

            if self.scaler:
                X = self.scaler.transform(X)

            # Predict probability
            proba = self.model.predict_proba(X)[0]
            win_idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else 0
            win_prob = proba[win_idx]

            # Calculate confidence adjustment: +10 if very likely, -15 if very unlikely
            if win_prob >= 0.75:
                adjustment = int((win_prob - 0.5) * 40)  # +10 to +20
            elif win_prob <= 0.35:
                adjustment = -int((0.5 - win_prob) * 30)  # -5 to -15
            else:
                adjustment = 0

            return {
                "win_probability": round(win_prob, 3),
                "confidence_adjustment": adjustment,
                "model_confidence": "HIGH" if abs(win_prob - 0.5) > 0.2 else "LOW",
            }
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return None

    def should_retrain(self) -> bool:
        """Check if enough new data exists to warrant retraining."""
        try:
            model_data = self.db.get_latest_ml_model("signal_predictor")
            last_count = model_data["total_samples"] if model_data else 0

            signals = self.db.get_closed_signals(limit=5000)
            current_count = len([s for s in signals if s.get("features")])

            new_samples = current_count - last_count
            return (current_count >= MIN_TRAINING_SAMPLES and
                    (new_samples >= RETRAIN_THRESHOLD or not self.is_loaded))
        except Exception:
            return False

    def train(self, force: bool = False) -> Optional[dict]:
        """Train/retrain the model on historical signal outcomes.
        Uses ONLY real signal data from database.
        Returns training metrics or None if insufficient data.
        """
        if not force and not self.should_retrain():
            logger.info("Not enough new data for retraining")
            return None

        # Import here to avoid slow startup
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report

        # Get signals with feature snapshots and known outcomes
        signals = self.db.get_signals_with_features(limit=2000)

        if len(signals) < MIN_TRAINING_SAMPLES:
            logger.warning(f"Need {MIN_TRAINING_SAMPLES} samples, have {len(signals)}")
            return None

        # Build training data from real signal outcomes
        X_list = []
        y_list = []

        for sig in signals:
            features = sig.get("features")
            outcome = sig.get("outcome", "")
            if not features or outcome in ("PENDING", "EXPIRED"):
                continue

            # Binary target: 1 = any target hit, 0 = SL hit
            label = 1 if outcome.startswith("T") else 0

            row = [features.get(f, 0) for f in self.feature_names]
            X_list.append(row)
            y_list.append(label)

        if len(X_list) < MIN_TRAINING_SAMPLES:
            logger.warning(f"Only {len(X_list)} valid samples after filtering")
            return None

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"Training ML model on {len(X)} samples (wins={sum(y)}, losses={len(y)-sum(y)})")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train gradient boosting classifier
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42,
        )

        # Cross-validation if enough data
        if len(X) >= 30:
            cv_folds = min(5, len(X) // 6)
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring="accuracy")
            cv_accuracy = cv_scores.mean() * 100
            logger.info(f"CV Accuracy: {cv_accuracy:.1f}% ± {cv_scores.std() * 100:.1f}%")
        else:
            cv_accuracy = 0

        # Fit on full data
        model.fit(X_scaled, y)

        # Feature importances
        importances = dict(zip(self.feature_names, model.feature_importances_.tolist()))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

        # Training accuracy
        train_accuracy = model.score(X_scaled, y) * 100

        # Prediction report
        y_pred = model.predict(X_scaled)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)

        metrics = {
            "train_accuracy": round(train_accuracy, 1),
            "cv_accuracy": round(cv_accuracy, 1),
            "total_samples": len(X),
            "win_samples": int(sum(y)),
            "loss_samples": int(len(y) - sum(y)),
            "top_features": [(f, round(v, 4)) for f, v in top_features],
            "precision_win": round(report.get("1", {}).get("precision", 0) * 100, 1),
            "recall_win": round(report.get("1", {}).get("recall", 0) * 100, 1),
            "f1_win": round(report.get("1", {}).get("f1-score", 0) * 100, 1),
            "trained_at": datetime.utcnow().isoformat(),
        }

        # Serialize model + scaler
        state = {"model": model, "scaler": scaler}
        model_bytes = pickle.dumps(state)

        # Save to database
        self.db.save_ml_model(
            model_name="signal_predictor",
            model_data=model_bytes,
            feature_names=self.feature_names,
            accuracy=cv_accuracy if cv_accuracy > 0 else train_accuracy,
            total_samples=len(X),
            metrics=metrics,
        )

        # Update instance
        self.model = model
        self.scaler = scaler
        self.is_loaded = True

        logger.info(
            f"ML model trained — accuracy: {metrics['cv_accuracy'] or metrics['train_accuracy']}%, "
            f"samples: {len(X)}, top feature: {top_features[0][0]}"
        )
        return metrics

    def get_model_info(self) -> dict:
        """Get info about current model state."""
        model_data = self.db.get_latest_ml_model("signal_predictor")
        if not model_data:
            return {"status": "NOT_TRAINED", "message": "ML modeli henüz eğitilmedi"}

        return {
            "status": "ACTIVE" if self.is_loaded else "INACTIVE",
            "accuracy": model_data.get("accuracy", 0),
            "total_samples": model_data.get("total_samples", 0),
            "trained_at": model_data.get("trained_at", ""),
            "metrics": model_data.get("metrics", {}),
        }
