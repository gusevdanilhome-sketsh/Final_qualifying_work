#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль конфигурации параметров системы.
"""

import numpy as np

C_LIGHT = 3e8

LINE_PARAMS = {
    'width': 0.01,
    'height': 0.0005,
    'thickness': 0.000035,
    'epsilon_r': 4.5,
    'length': 0.1,
    'z0_nom': 50.0
}

PROBE_PARAMS = {
    'a': 0.002,
    'positions': [0.02, 0.04, 0.06, 0.08]
}

FREQ_PARAMS = {
    'frequencies': np.linspace(1e9, 8e9, 8),
    'power': 1.0
}

NOISE_PARAMS = {
    'snr_db': 40,
    'seed': 42
}

DATASET_PARAMS = {
    'n_samples_per_class': 380,
    'classes': 5
}