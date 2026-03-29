#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Классификатор Random Forest (рекомендован в VKR.docx).
Оптимизирован для достижения точности ≥0.8.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from classifiers.base_classifier import classifier_t

class random_forest_classifier_t(classifier_t):
    """Реализация классификатора случайный лес."""
    
    def __init__(self):
        # Оптимизированные параметры согласно VKR.docx раздел 1.3.6
        self.model = RandomForestClassifier(
            n_estimators=300,           # Количество деревьев
            max_depth=25,               # Максимальная глубина
            min_samples_leaf=2,         # Минимум образцов в листе
            max_features='sqrt',        # Количество признаков для разбиения
            random_state=42,            # Фиксированный seed
            n_jobs=-1,                  # Использование всех ядер
            class_weight='balanced',    # Балансировка классов
            bootstrap=True,             # Бутстреп-выборка
            oob_score=True              # OOB оценка
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_name(self) -> str:
        return "Random Forest"
    
    def get_feature_importance(self) -> np.ndarray:
        """Получение важности признаков."""
        return self.model.feature_importances_