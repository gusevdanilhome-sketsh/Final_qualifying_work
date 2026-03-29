#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор синтетических данных с учетом шумовой модели.
Основано на разделе 4 VKR_V2.docx.
"""

import numpy as np
import pandas as pd
from models.microstrip_line import microstrip_line_t
from hardware.measurement_system import measurement_system_t
from config.parameters import LINE_PARAMS, PROBE_PARAMS, FREQ_PARAMS, NOISE_PARAMS, DATASET_PARAMS

class data_generator_t:
    """Генерация выборки для обучения классификаторов."""
    
    def __init__(self):
        self.line_nominal = microstrip_line_t(
            LINE_PARAMS['width'], LINE_PARAMS['height'], 
            LINE_PARAMS['thickness'], LINE_PARAMS['epsilon_r']
        )
        self.measurement_system = measurement_system_t(PROBE_PARAMS['a'])
        np.random.seed(NOISE_PARAMS['seed'])

    def _create_defect_line(self, defect_type: int, severity: float) -> microstrip_line_t:
        """Создание модели линии с дефектом."""
        w = LINE_PARAMS['width']
        h = LINE_PARAMS['height']
        t = LINE_PARAMS['thickness']
        eps = LINE_PARAMS['epsilon_r']
        
        if defect_type == 1: # Утонение высоты проводника
            t *= severity
        elif defect_type == 2: # Утонение ширины проводника
            w *= severity
        elif defect_type == 3: # Утонение подложки
            h *= severity
        elif defect_type == 4: # Изменение диэлектрической проницаемости
            eps *= severity
            
        return microstrip_line_t(w, h, t, eps)

    def generate_sample(self, defect_type: int, defect_pos: float) -> np.ndarray:
        """Генерация одного вектора признаков."""
        features = []
        severity = np.random.uniform(0.7, 0.9) if defect_type > 0 else 1.0
        
        line_defect = self._create_defect_line(defect_type, severity)
        z_load = line_defect.z0 
        gamma = self.line_nominal.get_reflection_coefficient(z_load)
        
        if defect_type == 0:
            gamma = 0
            line_defect = self.line_nominal

        for freq in FREQ_PARAMS['frequencies']:
            head_center = defect_pos 
            iq_features = self.measurement_system.process_signal(
                line_defect, freq, head_center, defect_pos, gamma
            )
            features.extend(iq_features)
            
        features = np.array(features)
        noise_level = self._calculate_noise_level(features)
        noise = np.random.normal(0, noise_level, features.shape)
        
        return features + noise

    def _calculate_noise_level(self, signal: np.ndarray) -> float:
        """Расчет уровня шума исходя из SNR."""
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(NOISE_PARAMS['snr_db'] / 10)
        noise_power = signal_power / snr_linear
        return np.sqrt(noise_power)

    def generate_dataset(self) -> pd.DataFrame:
        """Генерация полного набора данных."""
        data = []
        labels = []
        
        samples_per_class = DATASET_PARAMS['n_samples_per_class']
        
        for cls in range(DATASET_PARAMS['classes']):
            for _ in range(samples_per_class):
                pos = np.random.choice(PROBE_PARAMS['positions'])
                sample = self.generate_sample(cls, pos)
                data.append(sample)
                labels.append(cls)
                
        df = pd.DataFrame(data)
        df['label'] = labels
        return df