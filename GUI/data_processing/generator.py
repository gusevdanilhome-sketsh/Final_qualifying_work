#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор синтетических данных.
Реализует шаг сканирования 0.01 мм и порядок дефектов по сид.
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
    
    def __init__(self, seed=None):
        # Установка сида для воспроизводимости
        self.seed = seed if seed is not None else NOISE_PARAMS['seed']
        
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
        """Генерация названий столбцов с отображением частоты, составляющей и канала."""
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

    def _generate_defect_sequence_from_seed(self) -> list:
        """
        Генерация последовательности дефектов на основе сида.
        Чётные сегменты (0,2,4,6,8) - без дефекта
        Нечётные сегменты (1,3,5,7,9) - с дефектом, тип определяется сидом
        
        Returns:
            Список типов дефектов для каждого сегмента
        """
        # Фиксируем сид для генерации последовательности
        rng = np.random.RandomState(self.seed)
        
        segment_sequence = []
        n_segments = DEFECT_PARAMS['n_segments']
        defect_types = DEFECT_PARAMS['defect_types'].copy()
        
        # Перемешиваем типы дефектов на основе сида
        rng.shuffle(defect_types)
        
        defect_idx = 0
        for seg_idx in range(n_segments):
            if seg_idx % 2 == 0:
                # Чётные сегменты - без дефекта
                segment_sequence.append(0)
            else:
                # Нечётные сегменты - с дефектом (тип определяется сидом)
                # Циклически используем перемешанные типы дефектов
                segment_sequence.append(defect_types[defect_idx % len(defect_types)])
                defect_idx += 1
        
        return segment_sequence

    def generate_dataset(self) -> pd.DataFrame:
        """
        Генерация полного набора данных.
        Шаг сканирования: 0.01 мм
        Последовательность: определяется сидом
        """
        # Фиксируем сид для генерации данных
        np.random.seed(self.seed)
        
        data = []
        labels = []
        positions = []
        severities = []
        segment_indices = []
        
        # Получаем последовательность сегментов на основе сида
        segment_sequence = self._generate_defect_sequence_from_seed()
        n_segments = len(segment_sequence)
        samples_per_segment = DEFECT_PARAMS['samples_per_segment']
        segment_length = DEFECT_PARAMS['segment_length']
        scanning_step = DEFECT_PARAMS['scanning_step']
        
        # Генерация данных для каждого сегмента
        for seg_idx in range(n_segments):
            defect_type = segment_sequence[seg_idx]
            
            for sample_idx in range(samples_per_segment):
                # Позиция внутри сегмента (шаг 0.01 мм)
                pos_in_segment = sample_idx * scanning_step
                position = seg_idx * segment_length + pos_in_segment + scanning_step
                
                severity = np.random.uniform(*DEFECT_PARAMS['severity_range']) if defect_type > 0 else 1.0
                sample = self.generate_sample(defect_type, position)
                
                data.append(sample)
                labels.append(defect_type)
                positions.append(position)
                severities.append(severity)
                segment_indices.append(seg_idx)
        
        df = pd.DataFrame(data)
        
        # Добавление названий столбцов
        feature_names = self._generate_column_names()
        df.columns = feature_names
        
        df['label'] = labels
        df['defect_position'] = positions
        df['defect_severity'] = severities
        df['segment_index'] = segment_indices
        
        return df

    def generate_scanning_dataset(self) -> pd.DataFrame:
        """
        Генерация данных сканирования линии с шагом 0.01 мм.
        Реализует последовательность согласно сид.
        """
        np.random.seed(self.seed)
        
        data = []
        labels = []
        positions = []
        segment_indices = []
        
        segment_sequence = self._generate_defect_sequence_from_seed()
        n_segments = len(segment_sequence)
        samples_per_segment = DEFECT_PARAMS['samples_per_segment']
        segment_length = DEFECT_PARAMS['segment_length']
        scanning_step = DEFECT_PARAMS['scanning_step']
        
        for seg_idx in range(n_segments):
            defect_type = segment_sequence[seg_idx]
            
            for sample_idx in range(samples_per_segment):
                pos_in_segment = sample_idx * scanning_step
                position = seg_idx * segment_length + pos_in_segment + scanning_step
                
                sample = self.generate_sample(defect_type, position)
                data.append(sample)
                labels.append(defect_type)
                positions.append(position)
                segment_indices.append(seg_idx)
        
        df = pd.DataFrame(data)
        feature_names = self._generate_column_names()
        df.columns = feature_names
        df['label'] = labels
        df['position'] = positions
        df['segment_index'] = segment_indices
        
        return df