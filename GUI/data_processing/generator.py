#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор синтетических данных.
Реализует точное распределение классов согласно таблице:
10мм без дефекта, 10мм дефект, 10мм без дефекта...
Сид определяет порядок типов дефектов в сегментах с дефектами.
"""

import numpy as np
import pandas as pd
from models.microstrip_line import microstrip_line_t
from hardware.measurement_system import measurement_system_t
from config.parameters import (
    LINE_PARAMS, PROBE_PARAMS, FREQ_PARAMS, 
    NOISE_PARAMS, DATASET_PARAMS, DEFECT_PARAMS, 
    CLASS_NAMES, SEGMENT_SEQUENCE
)

class data_generator_t:
    """Генерация выборки для обучения классификаторов."""
    
    def __init__(self, seed=None):
        # Установка сида для воспроизводимости
        self.seed = seed if seed is not None else NOISE_PARAMS['seed']
        np.random.seed(self.seed)
        
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
        Возвращает последовательность согласно таблице:
        [0, 1, 0, 2, 0, 1, 0, 3, 0, 4] или вариации в зависимости от seed.
        """
        np.random.seed(self.seed)
        
        # Базовая последовательность: чередование без дефекта и с дефектом
        # Сид определяет какие типы дефектов в сегментах с дефектами
        base_sequence = SEGMENT_SEQUENCE.copy()
        
        # Если seed отличается от стандартного, варьируем типы дефектов
        if self.seed != 42:
            # Перемешиваем типы дефектов для сегментов с дефектами
            defect_indices = [i for i, x in enumerate(base_sequence) if x != 0]
            defect_types = [base_sequence[i] for i in defect_indices]
            np.random.shuffle(defect_types)
            for idx, def_type in zip(defect_indices, defect_types):
                base_sequence[idx] = def_type
        
        return base_sequence

    def generate_dataset(self) -> pd.DataFrame:
        """
        Генерация полного набора данных согласно таблице распределения.
        Последовательность: 10мм без дефекта, 10мм дефект, 10мм без дефекта...
        """
        np.random.seed(self.seed)
        
        data = []
        labels = []
        positions = []
        severities = []
        segment_indices = []
        
        # Получаем последовательность сегментов
        segment_sequence = self._generate_defect_sequence_from_seed()
        n_segments = len(segment_sequence)
        samples_per_segment = DATASET_PARAMS['n_samples_per_segment']
        segment_length = DEFECT_PARAMS['segment_length']
        scanning_step = DEFECT_PARAMS['scanning_step']
        
        # Генерация данных для каждого сегмента
        for seg_idx in range(n_segments):
            defect_type = segment_sequence[seg_idx]
            
            for sample_idx in range(samples_per_segment):
                # Позиция внутри сегмента
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
        Генерация данных сканирования линии с шагом 0.1 мм.
        Реализует последовательность согласно таблице.
        """
        np.random.seed(self.seed)
        
        data = []
        labels = []
        positions = []
        segment_indices = []
        
        segment_sequence = self._generate_defect_sequence_from_seed()
        n_segments = len(segment_sequence)
        samples_per_segment = DATASET_PARAMS['n_samples_per_segment']
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