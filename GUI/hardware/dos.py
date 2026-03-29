#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модель диаграммообразующей схемы (ДОС).
"""

import numpy as np

class dos_t:
    """Модель диаграммообразующей схемы."""
    
    def apply_dos_matrix(self, electrode_voltages: np.ndarray) -> np.ndarray:
        """
        Применение матрицы рассеяния ДОС.
        Возвращает 3 канала: Sum, Diff_X, Diff_Y.
        """
        s_matrix = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5, 0.5]
        ])
        return np.dot(s_matrix, electrode_voltages)