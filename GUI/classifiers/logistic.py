#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Логистическая регрессия с оптимизированными параметрами.
"""

from sklearn.linear_model import LogisticRegression
from classifiers.base_classifier import classifier_t

class logistic_classifier_t(classifier_t):
    """Реализация логистической регрессии."""
    
    def __init__(self):
        # Убрано multi_class (устаревший параметр)
        self.model = LogisticRegression(
            max_iter=2000, 
            solver='lbfgs',
            C=1.0,
            random_state=42,
            class_weight='balanced'
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_name(self) -> str:
        return "Logistic Regression"