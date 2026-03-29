#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Линейный дискриминантный анализ.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .base_classifier import classifier_t

class lda_classifier_t(classifier_t):
    """Реализация линейного дискриминантного анализа."""
    
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_name(self) -> str:
        return "LDA"