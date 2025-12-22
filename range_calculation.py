# range_calculation_final_fixed.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Константы
c = 299792458.0  # скорость света, м/с
c_km = 299792.458  # скорость света, км/с

class RangeConverter:
    """
    Класс для преобразования наблюдаемых значений в физические величины
    """
    def __init__(self):
        # Коэффициенты преобразования для разных систем дальнометрии
        # Для систем дальнометрии с модулем M
        self.modulo_M = 2**(6 + 6)  # 4096 для n=6
        self.scale_factor = 1.0  # Будем определять из данных
        
    def convert_observable_to_range(self, observable_value, data_type_id=11):
        """
        Преобразует наблюдаемое значение в дальность в единицах дальности
        
        Для data_type_id = 11 (Two-way Range):
        - Значения могут быть в разных системах
        - Могут быть в диапазоне ~1.29e6 (возможно, это наносекунды или циклы)
        - Нужно преобразовать в стандартные единицы дальности
        """
        if data_type_id == 11:
            # Проверяем масштаб значений
            abs_value = abs(observable_value)
            
            # Вариант 1: Если значения большие (~1e6), возможно это наносекунды
            if abs_value > 1e5:
                # Преобразуем наносекунды в секунды, затем в единицы дальности
                # 1 наносекунда = 1e-9 секунд
                time_seconds = observable_value * 1e-9
                # Преобразуем время в единицы дальности через скорость света
                # F = C_range * f_T = 1 для упрощения
                range_units = time_seconds  # Упрощенно
                return range_units
            
            # Вариант 2: Если значения маленькие, это уже единицы дальности
            else:
                return observable_value
        
        return observable_value
    
    def range_units_to_km(self, range_units, freq_band='X'):
        """
        Преобразует единицы дальности в километры
        """
        # Коэффициенты преобразования зависят от системы дальнометрии
        # Для X-band с частотой 8.4 GHz
        
        if freq_band == 'X':
            # Типичное преобразование для систем дальнометрии NASA
            # 1 единица дальности ≈ c / (2 * f_T) метров
            f_T = 8.4e9  # Гц
            meters_per_unit = c / (2 * f_T)  # метров на единицу
            km_per_unit = meters_per_unit / 1000  # км на единицу
            
            return range_units * km_per_unit
        
        return range_units * 1.0  # По умолчанию
    
    def compute_range_from_light_time(self, light_time_seconds, freq_band='X'):
        """
        Вычисляет дальность из времени прохождения света
        """
        # Two-way расстояние: туда и обратно
        distance_km = light_time_seconds * c_km / 2
        
        # Преобразуем км в единицы дальности
        km_per_unit = self.range_units_to_km(1.0, freq_band)
        if km_per_unit > 0:
            range_units = distance_km / km_per_unit
        else:
            range_units = distance_km
        
        return range_units, distance_km


def analyze_calibration(df_results):
    """
    Анализирует данные для калибровки преобразования
    """
    print("\n" + "="*70)
    print("КАЛИБРОВОЧНЫЙ АНАЛИЗ")
    print("="*70)
    
    if 'observed_observable' not in df_results.columns or 'computed_distance_km' not in df_results.columns:
        print("Недостаточно данных для калибровки")
        return None
    
    # Берем подвыборку для калибровки
    sample_size = min(50, len(df_results))
    sample_indices = np.random.choice(len(df_results), sample_size, replace=False)
    
    print(f"Используем {sample_size} записей для калибровки")
    
    # Собираем пары (наблюдаемое значение, вычисленное расстояние)
    calibration_data = []
    
    for idx in sample_indices:
        row = df_results.iloc[idx]
        observed = row['observed_observable']
        computed_km = row['computed_distance_km']
        
        if not pd.isna(observed) and not pd.isna(computed_km):
            calibration_data.append((observed, computed_km))
    
    if len(calibration_data) < 10:
        print("Недостаточно данных для калибровки")
        return None
    
    observed_vals, computed_kms = zip(*calibration_data)
    observed_vals = np.array(observed_vals)
    computed_kms = np.array(computed_kms)
    
    # Пытаемся найти линейное преобразование: observed = a * computed_km + b
    # Или наоборот: computed_km = a * observed + b
    
    # Вариант 1: Линейная регрессия
    from scipy import stats
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(observed_vals, computed_kms)
    
    print(f"\nЛинейная регрессия (computed_km = a * observed + b):")
    print(f"  a (slope): {slope:.6e}")
    print(f"  b (intercept): {intercept:.3f}")
    print(f"  R²: {r_value**2:.6f}")
    print(f"  p-value: {p_value:.6e}")
    
    # Вычисляем преобразованное значение
    converter = RangeConverter()
    predicted_kms = slope * observed_vals + intercept
    
    # Вычисляем ошибки
    errors = computed_kms - predicted_kms
    rmse = np.sqrt(np.mean(errors**2))
    mean_abs_error = np.mean(np.abs(errors))
    
    print(f"\nОшибки преобразования:")
    print(f"  RMSE: {rmse:.3f} км")
    print(f"  Средняя абсолютная ошибка: {mean_abs_error:.3f} км")
    print(f"  Относительная ошибка: {rmse/np.mean(computed_kms)*100:.3f}%")
    
    # Проверяем разные гипотезы преобразования
    print(f"\nПроверка гипотез преобразования:")
    
    # Гипотеза 1: Наблюдаемое значение - это время в наносекундах
    time_ns = observed_vals
    distance_from_time = time_ns * 1e-9 * c_km  # ns -> s -> km (one-way)
    errors_time = computed_kms - distance_from_time
    rmse_time = np.sqrt(np.mean(errors_time**2))
    print(f"  1. Наблюдаемое = время (нс), RMSE: {rmse_time:.3f} км")
    
    # Гипотеза 2: Наблюдаемое значение - это two-way время в наносекундах
    distance_from_2way_time = time_ns * 1e-9 * c_km / 2  # two-way
    errors_2way_time = computed_kms - distance_from_2way_time
    rmse_2way_time = np.sqrt(np.mean(errors_2way_time**2))
    print(f"  2. Наблюдаемое = two-way время (нс), RMSE: {rmse_2way_time:.3f} км")
    
    # Гипотеза 3: Наблюдаемое значение уже в единицах дальности
    # Нужно найти масштабный коэффициент
    scale_factors = computed_kms / observed_vals
    scale_factor = np.median(scale_factors[np.isfinite(scale_factors)])
    distance_from_units = observed_vals * scale_factor
    errors_units = computed_kms - distance_from_units
    rmse_units = np.sqrt(np.mean(errors_units**2))
    print(f"  3. Наблюдаемое в единицах дальности, масштаб: {scale_factor:.6e}, RMSE: {rmse_units:.3f} км")
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'scale_factor': scale_factor,
        'best_hypothesis': 'time_ns' if rmse_time < rmse_2way_time and rmse_time < rmse_units else
                          '2way_time_ns' if rmse_2way_time < rmse_units else 'units'
    }


def apply_calibration(df_results, calibration_params):
    """
    Применяет калибровку к данным
    """
    converter = RangeConverter()
    
    calibrated_observed_km = []
    calibrated_observed_units = []
    
    for idx, row in df_results.iterrows():
        observed = row['observed_observable']
        
        if pd.isna(observed):
            calibrated_observed_km.append(np.nan)
            calibrated_observed_units.append(np.nan)
            continue
        
        # Применяем лучшее преобразование
        if calibration_params['best_hypothesis'] == 'time_ns':
            # Наблюдаемое = время в наносекундах (one-way)
            time_seconds = observed * 1e-9
            distance_km = time_seconds * c_km
            range_units = observed  # Время в нс как единицы
            
        elif calibration_params['best_hypothesis'] == '2way_time_ns':
            # Наблюдаемое = two-way время в наносекундах
            time_seconds = observed * 1e-9
            distance_km = time_seconds * c_km / 2  # two-way
            range_units = observed  # Время в нс как единицы
            
        else:  # 'units'
            # Наблюдаемое уже в единицах дальности
            scale_factor = calibration_params['scale_factor']
            distance_km = observed * scale_factor
            range_units = observed
        
        calibrated_observed_km.append(distance_km)
        calibrated_observed_units.append(range_units)
    
    df_results['calibrated_observed_km'] = calibrated_observed_km
    df_results['calibrated_observed_units'] = calibrated_observed_units
    
    # Вычисляем разницу после калибровки
    valid_mask = ~pd.isna(df_results['calibrated_observed_km']) & ~pd.isna(df_results['computed_distance_km'])
    df_results.loc[valid_mask, 'calibrated_difference_km'] = (
        df_results.loc[valid_mask, 'calibrated_observed_km'] - 
        df_results.loc[valid_mask, 'computed_distance_km']
    )
    df_results.loc[valid_mask, 'calibrated_relative_error'] = (
        df_results.loc[valid_mask, 'calibrated_difference_km'] / 
        df_results.loc[valid_mask, 'computed_distance_km'] * 100
    )
    
    return df_results


def create_detailed_analysis(df_results, calibration_params):
    """
    Создает детальный анализ с калиброванными данными
    """
    print("\n" + "="*70)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ С КАЛИБРОВКОЙ")
    print("="*70)
    
    # Статистика до калибровки
    print("\n1. СТАТИСТИКА ДО КАЛИБРОВКИ:")
    if 'range_difference_units' in df_results.columns:
        diff_data = df_results['range_difference_units'].dropna()
        if len(diff_data) > 0:
            print(f"   Разница (наблюдаемое - вычисленное):")
            print(f"     Mean: {diff_data.mean():.3f} ед.")
            print(f"     Std: {diff_data.std():.3f} ед.")
            print(f"     RMS: {np.sqrt(np.mean(diff_data**2)):.3f} ед.")
    
    # Статистика после калибровки
    print("\n2. СТАТИСТИКА ПОСЛЕ КАЛИБРОВКИ:")
    if 'calibrated_difference_km' in df_results.columns:
        diff_data = df_results['calibrated_difference_km'].dropna()
        if len(diff_data) > 0:
            print(f"   Разница по расстоянию:")
            print(f"     Mean: {diff_data.mean():.3f} км")
            print(f"     Std: {diff_data.std():.3f} км")
            print(f"     RMS: {np.sqrt(np.mean(diff_data**2)):.3f} км")
            
            # Относительная ошибка
            if 'calibrated_relative_error' in df_results.columns:
                rel_error = df_results['calibrated_relative_error'].dropna()
                if len(rel_error) > 0:
                    print(f"     Средняя относительная ошибка: {rel_error.abs().mean():.3f}%")
                    print(f"     Макс. относительная ошибка: {rel_error.abs().max():.3f}%")
    
    # Проверка физической корректности
    print("\n3. ФИЗИЧЕСКАЯ КОРРЕКТНОСТЬ:")
    
    # Сравниваем вычисленное и калиброванное наблюдаемое расстояние
    if 'computed_distance_km' in df_results.columns and 'calibrated_observed_km' in df_results.columns:
        computed_mean = df_results['computed_distance_km'].mean()
        observed_mean = df_results['calibrated_observed_km'].mean()
        
        print(f"   Среднее вычисленное расстояние: {computed_mean/1e6:.3f} млн км")
        print(f"   Среднее калиброванное наблюдаемое: {observed_mean/1e6:.3f} млн км")
        print(f"   Разница: {abs(computed_mean - observed_mean)/1e3:.1f} тыс. км")
        print(f"   Относительная разница: {abs(computed_mean - observed_mean)/computed_mean*100:.3f}%")
    
    # Время прохождения
    if 'computed_light_time_s' in df_results.columns:
        light_time_mean = df_results['computed_light_time_s'].mean()
        expected_2way_time = computed_mean * 2 / c_km if 'computed_distance_km' in df_results.columns else 0
        
        print(f"\n   Время прохождения сигнала:")
        print(f"     Вычисленное one-way: {light_time_mean:.3f} с")
        print(f"     Ожидаемое two-way: {expected_2way_time:.3f} с")
        print(f"     Соотношение: {expected_2way_time/light_time_mean:.3f} (должно быть ~2.0)")
    
    # Построение графиков калибровки
    try:
        plt.figure(figsize=(15, 10))
        
        # График 1: Сравнение вычисленного и калиброванного наблюдаемого расстояния
        if 'computed_distance_km' in df_results.columns and 'calibrated_observed_km' in df_results.columns:
            plt.subplot(2, 3, 1)
            valid_mask = ~pd.isna(df_results['computed_distance_km']) & ~pd.isna(df_results['calibrated_observed_km'])
            if valid_mask.any():
                plt.scatter(df_results.loc[valid_mask, 'computed_distance_km'] / 1e6,
                          df_results.loc[valid_mask, 'calibrated_observed_km'] / 1e6,
                          alpha=0.5, s=10)
                
                # Линия идеального соответствия
                min_val = min(df_results.loc[valid_mask, 'computed_distance_km'].min(),
                            df_results.loc[valid_mask, 'calibrated_observed_km'].min()) / 1e6
                max_val = max(df_results.loc[valid_mask, 'computed_distance_km'].max(),
                            df_results.loc[valid_mask, 'calibrated_observed_km'].max()) / 1e6
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                
                plt.xlabel('Вычисленное расстояние (млн км)')
                plt.ylabel('Калиброванное наблюдаемое (млн км)')
                plt.title('Сравнение после калибровки')
                plt.grid(True, alpha=0.3)
        
        # График 2: Остатки после калибровки
        if 'calibrated_difference_km' in df_results.columns:
            plt.subplot(2, 3, 2)
            diff_data = df_results['calibrated_difference_km'].dropna()
            if len(diff_data) > 0:
                plt.plot(df_results.loc[diff_data.index, 'record_time'], 
                        diff_data / 1e3, 'b.', alpha=0.6)
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                plt.xlabel('Время')
                plt.ylabel('Остатки (тыс. км)')
                plt.title('Остатки после калибровки')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
        
        # График 3: Гистограмма остатков
        if 'calibrated_difference_km' in df_results.columns:
            plt.subplot(2, 3, 3)
            diff_data = df_results['calibrated_difference_km'].dropna()
            if len(diff_data) > 0:
                plt.hist(diff_data / 1e3, bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel('Остатки (тыс. км)')
                plt.ylabel('Частота')
                plt.title('Распределение остатков')
                plt.grid(True, alpha=0.3)
        
        # График 4: Относительная ошибка
        if 'calibrated_relative_error' in df_results.columns:
            plt.subplot(2, 3, 4)
            rel_error = df_results['calibrated_relative_error'].dropna()
            if len(rel_error) > 0:
                plt.hist(rel_error, bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel('Относительная ошибка (%)')
                plt.ylabel('Частота')
                plt.title('Распределение относительной ошибки')
                plt.grid(True, alpha=0.3)
        
        # График 5: Наблюдаемое vs вычисленное во времени
        if 'calibrated_observed_km' in df_results.columns and 'computed_distance_km' in df_results.columns:
            plt.subplot(2, 3, 5)
            valid_mask = ~pd.isna(df_results['calibrated_observed_km']) & ~pd.isna(df_results['computed_distance_km'])
            if valid_mask.any():
                plt.plot(df_results.loc[valid_mask, 'record_time'],
                        df_results.loc[valid_mask, 'calibrated_observed_km'] / 1e6,
                        'b-', alpha=0.7, label='Наблюдаемое')
                plt.plot(df_results.loc[valid_mask, 'record_time'],
                        df_results.loc[valid_mask, 'computed_distance_km'] / 1e6,
                        'r-', alpha=0.7, label='Вычисленное')
                plt.xlabel('Время')
                plt.ylabel('Расстояние (млн км)')
                plt.title('Расстояние во времени')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
        
        # График 6: Преобразование наблюдаемых значений
        plt.subplot(2, 3, 6)
        if 'observed_observable' in df_results.columns and 'computed_distance_km' in df_results.columns:
            valid_mask = ~pd.isna(df_results['observed_observable']) & ~pd.isna(df_results['computed_distance_km'])
            if valid_mask.any():
                plt.scatter(df_results.loc[valid_mask, 'observed_observable'],
                          df_results.loc[valid_mask, 'computed_distance_km'] / 1e6,
                          alpha=0.5, s=10)
                plt.xlabel('Наблюдаемое значение')
                plt.ylabel('Вычисленное расстояние (млн км)')
                plt.title('Преобразование наблюдаемых значений')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('range_calibration_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n   Графики калибровки сохранены в range_calibration_analysis.png")
        
    except Exception as e:
        print(f"   Ошибка при построении графиков: {e}")


def create_final_report(df_results, calibration_params):
    """
    Создает итоговый отчет с результатами калибровки
    """
    print("\n" + "="*70)
    print("ИТОГОВЫЙ ОТЧЕТ С КАЛИБРОВКОЙ")
    print("="*70)
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("ОТЧЕТ ПО АНАЛИЗУ ДАННЫХ ДАЛЬНОМЕТРИИ MESSENGER")
    report_lines.append("="*70)
    report_lines.append(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Всего записей: {len(df_results)}")
    report_lines.append(f"Тип данных: Two-way Range (data_type_id = 11)")
    
    report_lines.append("\n" + "-"*70)
    report_lines.append("РЕЗУЛЬТАТЫ КАЛИБРОВКИ")
    report_lines.append("-"*70)
    
    report_lines.append(f"Лучшая гипотеза преобразования: {calibration_params['best_hypothesis']}")
    report_lines.append(f"R² линейной регрессии: {calibration_params['r_squared']:.6f}")
    
    if calibration_params['best_hypothesis'] == 'units':
        report_lines.append(f"Масштабный коэффициент: {calibration_params['scale_factor']:.6e}")
    
    report_lines.append("\n" + "-"*70)
    report_lines.append("СТАТИСТИКА ПОСЛЕ КАЛИБРОВКИ")
    report_lines.append("-"*70)
    
    if 'calibrated_difference_km' in df_results.columns:
        diff_data = df_results['calibrated_difference_km'].dropna()
        if len(diff_data) > 0:
            report_lines.append(f"Остатки расстояния (наблюдаемое - вычисленное):")
            report_lines.append(f"  Среднее: {diff_data.mean():.3f} км")
            report_lines.append(f"  Стандартное отклонение: {diff_data.std():.3f} км")
            report_lines.append(f"  RMS: {np.sqrt(np.mean(diff_data**2)):.3f} км")
            
            # В процентах от среднего расстояния
            if 'computed_distance_km' in df_results.columns:
                mean_distance = df_results['computed_distance_km'].mean()
                report_lines.append(f"  RMS относительно среднего расстояния: {np.sqrt(np.mean(diff_data**2))/mean_distance*100:.3f}%")
    
    if 'calibrated_relative_error' in df_results.columns:
        rel_error = df_results['calibrated_relative_error'].dropna()
        if len(rel_error) > 0:
            report_lines.append(f"\nОтносительная ошибка:")
            report_lines.append(f"  Средняя абсолютная: {rel_error.abs().mean():.3f}%")
            report_lines.append(f"  Максимальная: {rel_error.abs().max():.3f}%")
    
    report_lines.append("\n" + "-"*70)
    report_lines.append("ФИЗИЧЕСКИЕ ПАРАМЕТРЫ")
    report_lines.append("-"*70)
    
    if 'computed_distance_km' in df_results.columns:
        dist_data = df_results['computed_distance_km']
        report_lines.append(f"Расстояние Земля-Меркурий:")
        report_lines.append(f"  Среднее: {dist_data.mean()/1e6:.3f} млн км")
        report_lines.append(f"  Минимальное: {dist_data.min()/1e6:.3f} млн км")
        report_lines.append(f"  Максимальное: {dist_data.max()/1e6:.3f} млн км")
        report_lines.append(f"  Диапазон: {(dist_data.max() - dist_data.min())/1e6:.3f} млн км")
    
    if 'computed_light_time_s' in df_results.columns:
        time_data = df_results['computed_light_time_s']
        report_lines.append(f"\nВремя прохождения сигнала (one-way):")
        report_lines.append(f"  Среднее: {time_data.mean():.3f} с")
        report_lines.append(f"  Минимальное: {time_data.min():.3f} с")
        report_lines.append(f"  Максимальное: {time_data.max():.3f} с")
        
        # Проверка two-way времени
        if 'computed_distance_km' in df_results.columns:
            expected_2way_time = dist_data.mean() * 2 / c_km
            actual_2way_time = time_data.mean() * 2  # Предполагаем симметричность
            report_lines.append(f"\nПроверка two-way времени:")
            report_lines.append(f"  Ожидаемое: {expected_2way_time:.3f} с")
            report_lines.append(f"  Фактическое (2 × one-way): {actual_2way_time:.3f} с")
            report_lines.append(f"  Разница: {abs(expected_2way_time - actual_2way_time):.6f} с")
    
    report_lines.append("\n" + "-"*70)
    report_lines.append("ВЫВОДЫ")
    report_lines.append("-"*70)
    
    if 'calibrated_relative_error' in df_results.columns:
        mean_rel_error = df_results['calibrated_relative_error'].abs().mean()
        if mean_rel_error < 1.0:
            report_lines.append("✓ Качество калибровки: ОТЛИЧНОЕ")
            report_lines.append(f"  Средняя ошибка всего {mean_rel_error:.3f}%")
        elif mean_rel_error < 5.0:
            report_lines.append("✓ Качество калибровки: ХОРОШЕЕ")
            report_lines.append(f"  Средняя ошибка {mean_rel_error:.3f}%")
        elif mean_rel_error < 10.0:
            report_lines.append("⚠ Качество калибровки: УДОВЛЕТВОРИТЕЛЬНОЕ")
            report_lines.append(f"  Средняя ошибка {mean_rel_error:.3f}% - требуется проверка")
        else:
            report_lines.append("✗ Качество калибровки: НИЗКОЕ")
            report_lines.append(f"  Средняя ошибка {mean_rel_error:.3f}% - требуется перекалибровка")
    
    report_lines.append("\n" + "-"*70)
    report_lines.append("РЕКОМЕНДАЦИИ")
    report_lines.append("-"*70)
    
    report_lines.append("1. Проверить документацию по формату ODF данных MESSENGER")
    report_lines.append("2. Уточнить единицы измерения для full_observable при data_type_id = 11")
    report_lines.append("3. При необходимости скорректировать коэффициенты преобразования")
    report_lines.append("4. Проверить наличие систематических ошибок во времени")
    
    report_lines.append("\n" + "="*70)
    
    # Сохраняем отчет
    report_text = "\n".join(report_lines)
    with open('range_final_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nИтоговый отчет сохранен в range_final_report.txt")


def main():
    """
    Главная функция
    """
    print("="*70)
    print("СИСТЕМА КАЛИБРОВКИ И АНАЛИЗА ДАННЫХ ДАЛЬНОМЕТРИИ")
    print("="*70)
    
    # Загружаем результаты предыдущего анализа
    input_file = 'range_complete_analysis.csv'
    
    print(f"\n1. Загрузка данных из {input_file}...")
    
    try:
        df_results = pd.read_csv(input_file)
        
        if 'record_time' in df_results.columns:
            df_results['record_time'] = pd.to_datetime(df_results['record_time'])
        
        print(f"   Загружено {len(df_results)} записей")
        
    except Exception as e:
        print(f"   Ошибка загрузки: {e}")
        return
    
    # Анализируем данные для калибровки
    print("\n2. Анализ для калибровки...")
    calibration_params = analyze_calibration(df_results)
    
    if calibration_params is None:
        print("   Не удалось выполнить калибровку")
        return
    
    # Применяем калибровку
    print("\n3. Применение калибровки...")
    df_calibrated = apply_calibration(df_results, calibration_params)
    
    # Сохраняем калиброванные данные
    output_file = 'range_calibrated_results.csv'
    df_calibrated.to_csv(output_file, index=False)
    print(f"   Калиброванные данные сохранены в {output_file}")
    
    # Детальный анализ
    print("\n4. Детальный анализ...")
    create_detailed_analysis(df_calibrated, calibration_params)
    
    # Итоговый отчет
    print("\n5. Создание итогового отчета...")
    create_final_report(df_calibrated, calibration_params)
    
    print("\n" + "="*70)
    print("ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО")
    print("="*70)
    
    print("\nСОЗДАННЫЕ ФАЙЛЫ:")
    print("1. range_calibrated_results.csv - калиброванные данные")
    print("2. range_calibration_analysis.png - графики калибровки")
    print("3. range_final_report.txt - итоговый отчет")


if __name__ == "__main__":
    main()