#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модель измерительной системы: зонд, ДОС, детектор.
Основано на разделах 1.2 VKR.docx и 3 VKR_V2.docx.
"""

import numpy as np
from models.microstrip_line import microstrip_line_t

class probe_t:
    """Модель измерительной головки с четырьмя электродами."""
    
    def __init__(self, side_size: float):
        self.a = side_size
        # Координаты электродов относительно центра
        self.offsets = np.array([
            [-self.a/2, -self.a/2],
            [ self.a/2, -self.a/2],
            [ self.a/2,  self.a/2],
            [-self.a/2,  self.a/2]
        ])

    def get_electrode_voltages(self, line: microstrip_line_t, frequency: float, 
                               head_center: float, defect_pos: float, 
                               gamma: complex) -> np.ndarray:
        """Расчет напряжений на 4 электродах."""
        voltages = np.zeros(4, dtype=complex)
        x_coords = np.array([
            head_center - self.a/2,
            head_center + self.a/2,
            head_center + self.a/2,
            head_center - self.a/2
        ])
        
        for i, x in enumerate(x_coords):
            voltages[i] = line.get_voltage_distribution(frequency, x, defect_pos, gamma)
            
        return voltages

class dos_t:
    """Модель диаграммообразующей схемы (ДОС)."""
    
    def apply_dos_matrix(self, electrode_voltages: np.ndarray) -> np.ndarray:
        """
        Применение матрицы рассеяния ДОС (формула 14 VKR_V2, 17 VKR).
        Возвращает 3 канала: Sum, Diff_X, Diff_Y.
        """
        s_matrix = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5, 0.5]
        ])
        return np.dot(s_matrix, electrode_voltages)

class detector_t:
    """Модель квадратурного детектора."""
    
    def quadrature_detection(self, signals: np.ndarray) -> np.ndarray:
        """
        Квадратурное детектирование (формулы 20-21 VKR_V2).
        Извлечение действительной (I) и мнимой (Q) частей.
        """
        features = []
        for sig in signals:
            features.append(np.real(sig))
            features.append(np.imag(sig))
        return np.array(features)

class measurement_system_t:
    """Объединяющий класс измерительной системы."""
    
    def __init__(self, probe_size: float):
        self.probe = probe_t(probe_size)
        self.dos = dos_t()
        self.detector = detector_t()

    def process_signal(self, line: microstrip_line_t, frequency: float, 
                       head_center: float, defect_pos: float, gamma: complex) -> np.ndarray:
        """Полный цикл обработки сигнала от электродов до признаков."""
        v_electrodes = self.probe.get_electrode_voltages(
            line, frequency, head_center, defect_pos, gamma
        )
        v_channels = self.dos.apply_dos_matrix(v_electrodes)
        iq_features = self.detector.quadrature_detection(v_channels)
        return iq_features