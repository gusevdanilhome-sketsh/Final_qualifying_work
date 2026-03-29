#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Объединяющая модель измерительной системы.
Основано на разделах 1.2 VKR.docx и 3 VKR_V2.docx.
"""

import numpy as np
from .probe import probe_t
from .dos import dos_t
from .detector import detector_t
from models.microstrip_line import microstrip_line_t

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