#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Математическая модель микрополосковой линии передачи.
Основано на формулах раздела 1.1 VKR.docx.
"""

import numpy as np
from config.parameters import C_LIGHT

class microstrip_line_t:
    """Класс моделирования параметров микрополосковой линии."""
    
    def __init__(self, width: float, height: float, thickness: float, epsilon_r: float):
        self.width = width
        self.height = height
        self.thickness = thickness
        self.epsilon_r = epsilon_r
        
        self.eff_width = self._calculate_effective_width()
        self.epsilon_eff = self._calculate_epsilon_eff()
        self.z0 = self._calculate_z0()

    def _calculate_effective_width(self) -> float:
        """Вычисление эффективной ширины проводника (формулы 6.4-6.6 VKR)."""
        if self.width / self.height > 1:
            delta_w = (self.thickness / np.pi) * (1 + np.log(2 * np.pi * self.width / self.thickness))
        else:
            delta_w = (self.thickness / np.pi) * (1 + np.log(4 * np.pi * self.width / self.thickness))
        return self.width + delta_w

    def _calculate_epsilon_eff(self) -> float:
        """Вычисление эффективной диэлектрической проницаемости (формула 6.3 VKR)."""
        w_eff_h = self.eff_width / self.height
        if w_eff_h > 1:
            epsilon_eff = (self.epsilon_r + 1) / 2 + (self.epsilon_r - 1) / 2 * (1 + 12 * self.height / self.eff_width)**(-0.5)
        else:
            epsilon_eff = (self.epsilon_r + 1) / 2 + (self.epsilon_r - 1) / 2 * ((1 + 12 * self.height / self.eff_width)**(-0.5) + 0.04 * (1 - w_eff_h)**2)
        return epsilon_eff

    def _calculate_z0(self) -> float:
        """Вычисление волнового сопротивления (формулы 6.1-6.2 VKR)."""
        w_eff_h = self.eff_width / self.height
        if w_eff_h > 1:
            z0 = (120 * np.pi) / (np.sqrt(self.epsilon_eff) * (w_eff_h + 1.393 + 0.667 * np.log(w_eff_h + 1.444)))
        else:
            z0 = (60 / np.sqrt(self.epsilon_eff)) * np.log(8 * self.height / self.eff_width + 0.25 * self.eff_width / self.height)
        return z0

    def get_phase_constant(self, frequency: float) -> float:
        """Вычисление фазовой постоянной beta."""
        omega = 2 * np.pi * frequency
        beta = omega * np.sqrt(self.epsilon_eff) / C_LIGHT
        return beta

    def get_reflection_coefficient(self, z_load: complex) -> complex:
        """Вычисление коэффициента отражения Gamma (формула 12 VKR)."""
        return (z_load - self.z0) / (z_load + self.z0)

    def get_voltage_distribution(self, frequency: float, x: float, 
                                 defect_pos: float, gamma: complex) -> complex:
        """
        Вычисление комплексной амплитуды напряжения в точке x.
        Учитывает падающую и отраженную волны (формулы 13, 15 VKR).
        """
        beta = self.get_phase_constant(frequency)
        u_inc = np.exp(-1j * beta * x)
        
        if x <= defect_pos:
            u_ref = gamma * np.exp(1j * beta * (x - 2 * defect_pos))
        else:
            u_ref = 0
            
        return u_inc + u_ref