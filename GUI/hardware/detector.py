#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модель квадратурного детектора.
"""

import numpy as np

class detector_t:
    """Модель квадратурного детектора."""
    
    def quadrature_detection(self, signals: np.ndarray) -> np.ndarray:
        """
        Квадратурное детектирование.
        Извлечение действительной (I) и мнимой (Q) частей.
        """
        features = []
        for sig in signals:
            features.append(np.real(sig))
            features.append(np.imag(sig))
        return np.array(features)