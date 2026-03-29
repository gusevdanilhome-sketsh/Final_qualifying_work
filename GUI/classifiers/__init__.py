#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Инициализация пакета классификаторов.
"""

from .base_classifier import classifier_t
from .bayesian import bayesian_classifier_t
from .lda import lda_classifier_t
from .logistic import logistic_classifier_t

__all__ = [
    "classifier_t", 
    "bayesian_classifier_t", 
    "lda_classifier_t", 
    "logistic_classifier_t"
]