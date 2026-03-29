#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль построения карт дефектов микрополосковой линии.
Обновлено для шага 0.01 мм и последовательности по сид.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
from config.parameters import (
    OUTPUT_PARAMS, LINE_PARAMS, DEFECT_PARAMS, 
    CLASS_NAMES, CLASS_COLORS, FREQ_PARAMS
)

class defect_map_plotter_t:
    """Класс для построения карт дефектов."""
    
    def __init__(self):
        self.figures_dir = OUTPUT_PARAMS['figures_dir']
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Настройка шрифтов
        font_paths = ['C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
        for path in font_paths:
            if os.path.exists(path):
                font_manager.fontManager.addfont(path)
                plt.rcParams['font.family'] = 'Arial' if 'arial' in path else 'DejaVu Sans'
                plt.rcParams['axes.unicode_minus'] = False
                return

    def plot_defect_maps_by_frequency(self, data_df: pd.DataFrame, y_true: np.ndarray, 
                                       y_pred: np.ndarray) -> list:
        """Построение карт дефектов для каждой частоты отдельно."""
        n_freq = len(FREQ_PARAMS['frequencies'])
        filepaths = []
        
        line_length = LINE_PARAMS['length']
        line_width = LINE_PARAMS['width']
        segment_length = DEFECT_PARAMS['segment_length']
        
        if 'defect_position' in data_df.columns:
            positions = data_df['defect_position'].values
        elif 'position' in data_df.columns:
            positions = data_df['position'].values
        else:
            positions = np.linspace(0, line_length, len(y_true))
        
        for freq_idx in range(n_freq):
            freq = FREQ_PARAMS['frequencies'][freq_idx]
            freq_ghz = freq / 1e9
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            ax.add_patch(plt.Rectangle((0, -line_width/2), line_length, line_width, 
                                        fill=True, color='lightgray', alpha=0.5, label='МПЛ'))
            
            # Отображение границ сегментов (1 мм)
            n_segments = DEFECT_PARAMS['n_segments']
            for seg_idx in range(n_segments):
                seg_start = seg_idx * segment_length
                ax.axvline(seg_start, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            # Прореживание точек для читаемости (шаг 0.01 мм = много точек)
            step = max(1, len(y_true) // 200)
            
            for i in range(0, len(y_true), step):
                true_label = int(y_true[i])
                pred_label = int(y_pred[i])
                pos = positions[i]
                
                if true_label == 0:
                    color = 'gray'
                    marker = '.'
                else:
                    is_error = int(true_label != pred_label)
                    color = 'red' if is_error else CLASS_COLORS[true_label]
                    marker = 'x' if is_error else 'o'
                
                ax.plot(pos, 0, marker, color=color, markersize=6, alpha=0.7)
            
            ax.set_xlabel('Продольная координата, м')
            ax.set_ylabel('Поперечная координата, м')
            ax.set_title(f'Карта дефектов МПЛ на частоте {freq_ghz:.1f} ГГц\n(Шаг сканирования: 0.01 мм)')
            ax.set_xlim(0, line_length)
            ax.set_ylim(-line_width*2, line_width*2)
            ax.grid(True, alpha=0.3)
            
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Верно'),
                Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=10, label='Ошибка'),
                Line2D([0], [0], marker='.', color='w', markerfacecolor='gray', markersize=10, label='Без дефекта')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            filepath = os.path.join(self.figures_dir, f'defect_map_{freq_ghz:.1f}GHz.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            filepaths.append(filepath)
        
        return filepaths

    def plot_defect_map_combined(self, data_df: pd.DataFrame, y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> str:
        """Построение сводной карты дефектов по всем частотам."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        line_length = LINE_PARAMS['length']
        line_width = LINE_PARAMS['width']
        segment_length = DEFECT_PARAMS['segment_length']
        
        if 'defect_position' in data_df.columns:
            positions = data_df['defect_position'].values
        elif 'position' in data_df.columns:
            positions = data_df['position'].values
        else:
            positions = np.linspace(0, line_length, len(y_true))
        
        ax.add_patch(plt.Rectangle((0, -line_width/2), line_length, line_width, 
                                    fill=True, color='lightgray', alpha=0.5, label='МПЛ'))
        
        n_segments = DEFECT_PARAMS['n_segments']
        for seg_idx in range(n_segments):
            seg_start = seg_idx * segment_length
            ax.axvline(seg_start, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        step = max(1, len(y_true) // 200)
        
        for i in range(0, len(y_true), step):
            true_label = int(y_true[i])
            pred_label = int(y_pred[i])
            pos = positions[i]
            
            if true_label == 0:
                color = 'gray'
                marker = '.'
            else:
                is_error = int(true_label != pred_label)
                color = 'red' if is_error else CLASS_COLORS[true_label]
                marker = 'x' if is_error else 'o'
            
            ax.plot(pos, 0, marker, color=color, markersize=6, alpha=0.7)
        
        ax.set_xlabel('Продольная координата, м')
        ax.set_ylabel('Поперечная координата, м')
        ax.set_title('Сводная карта дефектов МПЛ по всем частотам\n(Шаг сканирования: 0.01 мм)')
        ax.set_xlim(0, line_length)
        ax.set_ylim(-line_width*2, line_width*2)
        ax.grid(True, alpha=0.3)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Верно'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=10, label='Ошибка'),
            Line2D([0], [0], marker='.', color='w', markerfacecolor='gray', markersize=10, label='Без дефекта')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, 'defect_map_combined.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath

    def plot_prediction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Построение графика точности по классам."""
        from sklearn.metrics import classification_report
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
        
        precisions = [report_dict[name]['precision'] for name in CLASS_NAMES]
        recalls = [report_dict[name]['recall'] for name in CLASS_NAMES]
        f1_scores = [report_dict[name]['f1-score'] for name in CLASS_NAMES]
        
        x = np.arange(len(CLASS_NAMES))
        width = 0.25
        
        ax.bar(x - width, precisions, width, label='Precision', color='blue', alpha=0.7)
        ax.bar(x, recalls, width, label='Recall', color='green', alpha=0.7)
        ax.bar(x + width, f1_scores, width, label='F1-Score', color='red', alpha=0.7)
        
        ax.set_xlabel('Класс дефекта')
        ax.set_ylabel('Значение метрики')
        ax.set_title('Метрики качества классификации по классам')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, 'accuracy_by_class.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath

    def plot_segment_sequence(self, data_df: pd.DataFrame, segment_sequence: list) -> str:
        """
        Построение диаграммы последовательности сегментов.
        Отображает порядок дефектов, определяемый сидом.
        """
        fig, ax = plt.subplots(figsize=(14, 4))
        
        line_length = LINE_PARAMS['length']
        segment_length = DEFECT_PARAMS['segment_length']
        n_segments = len(segment_sequence)
        
        for seg_idx in range(n_segments):
            seg_start = seg_idx * segment_length
            has_defect = segment_sequence[seg_idx] != 0
            color = 'lightgreen' if not has_defect else CLASS_COLORS[segment_sequence[seg_idx]]
            label = 'Без дефекта' if not has_defect else CLASS_NAMES[segment_sequence[seg_idx]]
            
            ax.add_patch(plt.Rectangle((seg_start, 0), segment_length, 1, 
                                        fill=True, color=color, alpha=0.7,
                                        label=label if seg_idx < 2 else ""))
            ax.text(seg_start + segment_length/2, 0.5, f'Сегмент {seg_idx+1}\n{CLASS_NAMES[segment_sequence[seg_idx]]}', 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Продольная координата, м')
        ax.set_ylabel('')
        ax.set_title('Последовательность сегментов микрополосковой линии\n(Порядок дефектов определяется сидом, шаг сканирования: 0.01 мм)')
        ax.set_xlim(0, line_length)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, 'segment_sequence.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath