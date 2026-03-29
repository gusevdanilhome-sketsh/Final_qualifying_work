#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Классификатор Random Forest (рекомендован в VKR.docx).
"""

from sklearn.ensemble import RandomForestClassifier
from classifiers.base_classifier import classifier_t

class random_forest_classifier_t(classifier_t):
    """Реализация классификатора случайный лес."""
    
    def __init__(self):
        # Оптимизированные параметры согласно VKR.docx
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Для борьбы с дисбалансом классов
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_name(self) -> str:
        return "Random Forest"