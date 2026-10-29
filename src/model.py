"""
Machine Learning & NLP classification pipeline for Fake News Detection.
Includes TF-IDF Vectorization, multiple model training, evaluation, calibration,
ensemble predictions, and feature-importance explainability.
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Tuple, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from src.preprocessor import clean_text, combine_headline_and_body

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class FakeNewsModelManager:
    def __init__(self, model_type: str = "PassiveAggressive"):
        """
        model_type options: 'PassiveAggressive', 'LogisticRegression', 'NaiveBayes', 'LinearSVC', 'Ensemble'
        """
        self.model_type = model_type
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.models: Dict[str, Any] = {}
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}
        
    def _create_base_model(self, name: str):
        if name == "PassiveAggressive":
            base = PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=0.5)
            # Calibrate so we have smooth, probabilistic outputs
            return CalibratedClassifierCV(estimator=base, cv=3)
        elif name == "LogisticRegression":
            return LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        elif name == "NaiveBayes":
            return MultinomialNB(alpha=0.5)
        elif name == "LinearSVC":
            base = LinearSVC(max_iter=2000, random_state=42, C=0.5)
            return CalibratedClassifierCV(estimator=base, cv=3)
        else:
            return LogisticRegression(max_iter=1000, random_state=42)

    def train_all(self, df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
        """
        Trains all models on provided DataFrame.
        DataFrame must contain 'title', 'text', and 'label' ('REAL' / 'FAKE' or 0 / 1).
        """
        # Ensure processed text column
        if "clean_content" not in df.columns:
            df = df.copy()
            df["clean_content"] = df.apply(
                lambda row: combine_headline_and_body(str(row.get("title", "")), str(row.get("text", ""))),
                axis=1
            )
            
        # Target labels
        if "label_num" in df.columns:
            y = df["label_num"].values
        else:
            # Map strings to 0 (REAL) and 1 (FAKE)
            y = df["label"].astype(str).str.strip().str.upper().map(lambda v: 1 if "FAKE" in v or v == "1" else 0).values

        X = df["clean_content"].values
        
        # Train / Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # TF-IDF Vectorization with unigram and bigram features
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=4000,
            sublinear_tf=True
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        results = {}
        model_names = ["PassiveAggressive", "LogisticRegression", "NaiveBayes", "LinearSVC"]
        
        for name in model_names:
            model = self._create_base_model(name)
            model.fit(X_train_vec, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test_vec)
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            self.models[name] = model
            results[name] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "confusion_matrix": cm,
                "test_samples": len(y_test)
            }
            
        # Also fit an uncalibrated linear model for feature explainability coefficients
        self.explain_model = LogisticRegression(max_iter=1000, random_state=42)
        self.explain_model.fit(X_train_vec, y_train)
        
        self.is_trained = True
        self.metrics = results
        return results

    def predict(self, title: str, text: str, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs prediction on raw input headline and text body.
        Returns label ('REAL' or 'FAKE'), confidence percentages, per-model scores, and explainability tokens.
        """
        if not self.is_trained or self.vectorizer is None:
            raise ValueError("Model is not trained yet. Call train_all() first.")
            
        chosen_model = model_type or self.model_type
        clean_input = combine_headline_and_body(title, text)
        vec_input = self.vectorizer.transform([clean_input])
        
        model_predictions = {}
        probs = []
        
        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(vec_input)[0]  # [prob_real, prob_fake]
                fake_prob = float(prob[1])
            else:
                pred = model.predict(vec_input)[0]
                fake_prob = 1.0 if pred == 1 else 0.0
                
            model_predictions[name] = {
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(1.0 - fake_prob, 4),
                "verdict": "FAKE" if fake_prob >= 0.5 else "REAL"
            }
            probs.append(fake_prob)
            
        # Primary prediction based on chosen model or ensemble average
        if chosen_model == "Ensemble":
            avg_fake_prob = float(np.mean(probs))
            primary_fake_prob = avg_fake_prob
        elif chosen_model in self.models:
            primary_fake_prob = model_predictions[chosen_model]["fake_probability"]
        else:
            primary_fake_prob = model_predictions["PassiveAggressive"]["fake_probability"]
            
        verdict = "FAKE" if primary_fake_prob >= 0.5 else "REAL"
        confidence = primary_fake_prob if verdict == "FAKE" else (1.0 - primary_fake_prob)
        
        # Extract Explainability Features
        explanation = self._explain_prediction(vec_input, clean_input)
        
        return {
            "verdict": verdict,
            "confidence": round(confidence * 100, 2),
            "fake_probability_percent": round(primary_fake_prob * 100, 2),
            "real_probability_percent": round((1.0 - primary_fake_prob) * 100, 2),
            "chosen_model": chosen_model,
            "model_breakdown": model_predictions,
            "explanation": explanation
        }

    def _explain_prediction(self, vec_input, clean_input: str) -> Dict[str, Any]:
        """
        Finds which n-grams present in the text exerted the strongest pull toward FAKE or REAL.
        """
        if not hasattr(self, "explain_model") or self.explain_model is None or self.vectorizer is None:
            return {"fake_signals": [], "real_signals": []}
            
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        coefs = self.explain_model.coef_[0]
        
        # Get indices of non-zero entries in this document
        row = vec_input.tocoo()
        active_indices = row.col
        active_tfidf = row.data
        
        if len(active_indices) == 0:
            return {"fake_signals": [], "real_signals": []}
            
        contributions = []
        for idx, tfidf_val in zip(active_indices, active_tfidf):
            word = feature_names[idx]
            weight = coefs[idx] * tfidf_val
            contributions.append((word, float(weight)))
            
        # Sort by weight: positive = pushes FAKE, negative = pushes REAL
        fake_cues = sorted([c for c in contributions if c[1] > 0], key=lambda x: x[1], reverse=True)[:6]
        real_cues = sorted([c for c in contributions if c[1] < 0], key=lambda x: x[1])[:6]
        
        return {
            "fake_signals": [{"term": term, "weight": round(weight, 3)} for term, weight in fake_cues],
            "real_signals": [{"term": term, "weight": round(abs(weight), 3)} for term, weight in real_cues]
        }

    def save(self, filepath: str):
        joblib.dump({
            "vectorizer": self.vectorizer,
            "models": self.models,
            "explain_model": getattr(self, "explain_model", None),
            "metrics": self.metrics,
            "is_trained": self.is_trained
        }, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.vectorizer = data["vectorizer"]
        self.models = data["models"]
        self.explain_model = data.get("explain_model")
        self.metrics = data["metrics"]
        self.is_trained = data["is_trained"]
