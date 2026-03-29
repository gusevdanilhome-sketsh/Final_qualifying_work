#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Логистическая регрессия.
"""

from sklearn.linear_model import LogisticRegression
from .base_classifier import classifier_t

class logistic_classifier_t(classifier_t):
    """Реализация логистической регрессии."""
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, multi_class='multinomial')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_name(self) -> str:
        return "Logistic"