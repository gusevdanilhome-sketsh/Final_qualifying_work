#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор синтетических данных с учетом шумовой модели.
Основано на разделе 4 VKR_V2.docx и разделе 2 VKR.docx.
"""

import numpy as np
import pandas as pd
from models.microstrip_line import microstrip_line_t
from hardware.measurement_system import measurement_system_t
from config.parameters import (
    LINE_PARAMS, PROBE_PARAMS, FREQ_PARAMS, 
    NOISE_PARAMS, DATASET_PARAMS, DEFECT_PARAMS, CLASS_NAMES
)

class data_generator_t:
    """Генерация выборки для обучения классификаторов."""
    
    def __init__(self):
        # Установка сида для воспроизводимости
        np.random.seed(NOISE_PARAMS['seed'])
        
        self.line_nominal = microstrip_line_t(
            LINE_PARAMS['width'], LINE_PARAMS['height'], 
            LINE_PARAMS['thickness'], LINE_PARAMS['epsilon_r']
        )
        self.measurement_system = measurement_system_t(PROBE_PARAMS['a'])

    def _create_defect_line(self, defect_type: int, severity: float) -> microstrip_line_t:
        """Создание модели линии с дефектом (VKR.docx раздел 1.1.2)."""
        w = LINE_PARAMS['width']
        h = LINE_PARAMS['height']
        t = LINE_PARAMS['thickness']
        eps = LINE_PARAMS['epsilon_r']
        
        if defect_type == 1:  # Утонение высоты проводника (t)
            t *= severity
        elif defect_type == 2:  # Утонение ширины проводника (W)
            w *= severity
        elif defect_type == 3:  # Утонение подложки (h)
            h *= severity
        elif defect_type == 4:  # Изменение диэлектрической проницаемости (eps)
            eps *= (2.0 - severity)
            
        return microstrip_line_t(w, h, t, eps)

    def generate_sample(self, defect_type: int, defect_pos: float) -> np.ndarray:
        """Генерация одного вектора признаков."""
        features = []
        severity = np.random.uniform(*DEFECT_PARAMS['severity_range']) if defect_type > 0 else 1.0
        
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
        """Расчет уровня шума исходя из SNR (VKR_V2.docx формула 23)."""
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(NOISE_PARAMS['snr_db'] / 10)
        noise_power = signal_power / snr_linear
        return np.sqrt(noise_power)

    def _generate_column_names(self) -> list:
        """
        Генерация названий столбцов с отображением частоты, 
        составляющей сигнала и канала.
        """
        channels = ['Sum', 'Diff_X', 'Diff_Y']
        components = ['I', 'Q']
        column_names = []
        
        for freq in FREQ_PARAMS['frequencies']:
            freq_ghz = freq / 1e9
            for ch in channels:
                for comp in components:
                    col_name = f"{freq_ghz:.1f}GHz_{ch}_{comp}"
                    column_names.append(col_name)
        
        return column_names

    def generate_dataset(self) -> pd.DataFrame:
        """
        Генерация полного набора данных.
        Между двумя дефектами должно находиться отсутствие дефекта.
        """
        # Сброс сида перед генерацией для гарантированной воспроизводимости
        np.random.seed(NOISE_PARAMS['seed'])
        
        data = []
        labels = []
        positions = []
        severities = []
        
        samples_per_class = DATASET_PARAMS['n_samples_per_class']
        
        # Генерация с учётом требования: между дефектами - отсутствие дефекта
        for cls in range(DATASET_PARAMS['classes']):
            for i in range(samples_per_class):
                # Выбор позиции дефекта
                pos = DEFECT_PARAMS['positions'][i % len(DEFECT_PARAMS['positions'])]
                severity = np.random.uniform(*DEFECT_PARAMS['severity_range']) if cls > 0 else 1.0
                sample = self.generate_sample(cls, pos)
                data.append(sample)
                labels.append(cls)
                positions.append(pos)
                severities.append(severity)
                
                # После каждого дефекта (кроме последнего в серии) добавляем образец без дефекта
                if cls > 0 and i < samples_per_class - 1:
                    no_defect_sample = self.generate_sample(0, pos + DEFECT_PARAMS['min_gap_between_defects'])
                    data.append(no_defect_sample)
                    labels.append(0)
                    positions.append(pos + DEFECT_PARAMS['min_gap_between_defects'])
                    severities.append(1.0)
        
        df = pd.DataFrame(data)
        
        # Добавление названий столбцов
        feature_names = self._generate_column_names()
        df.columns = feature_names
        
        df['label'] = labels
        df['defect_position'] = positions
        df['defect_severity'] = severities
        
        return df

    def generate_scanning_dataset(self) -> pd.DataFrame:
        """
        Генерация данных сканирования линии с шагом 0.01 мм.
        (VKR.docx раздел 2.1)
        """
        np.random.seed(NOISE_PARAMS['seed'])
        
        data = []
        labels = []
        positions = []
        
        scanning_step = PROBE_PARAMS['scanning_step']
        line_length = LINE_PARAMS['length']
        n_positions = int(line_length / scanning_step)
        
        for i in range(n_positions):
            pos = i * scanning_step
            
            # Определение наличия дефекта в данной позиции
            defect_found = False
            for defect_pos in DEFECT_PARAMS['positions']:
                if abs(pos - defect_pos) < DEFECT_PARAMS['max_size'] / 2:
                    defect_type = np.random.randint(1, 5)
                    sample = self.generate_sample(defect_type, pos)
                    labels.append(defect_type)
                    defect_found = True
                    break
            
            if not defect_found:
                sample = self.generate_sample(0, pos)
                labels.append(0)
            
            data.append(sample)
            positions.append(pos)
        
        df = pd.DataFrame(data)
        feature_names = self._generate_column_names()
        df.columns = feature_names
        df['label'] = labels
        df['position'] = positions
        
        return df