#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Байесовский классификатор.
"""

from sklearn.naive_bayes import GaussianNB
from .base_classifier import classifier_t

class bayesian_classifier_t(classifier_t):
    """Реализация наивного байесовского классификатора."""
    
    def __init__(self):
        self.model = GaussianNB()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_name(self) -> str:
        return "Bayesian"