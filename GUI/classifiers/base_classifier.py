#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Базовый класс для всех классификаторов.
"""

from abc import ABC, abstractmethod

class classifier_t(ABC):
    """Абстрактный базовый класс классификатора."""
    
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass