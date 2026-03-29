#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модель измерительной головки (зонда).
"""

import numpy as np
from models.microstrip_line import microstrip_line_t

class probe_t:
    """Модель измерительной головки с четырьмя электродами."""
    
    def __init__(self, side_size: float):
        self.a = side_size
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