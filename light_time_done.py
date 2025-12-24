# light_time_done.py

import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import pandas as pd
from astropy.time import Time
import matplotlib.pyplot as plt
import os
import sys

# Константы
c = 299792458.0  # скорость света, м/с
c_km = 299792.458  # скорость света, км/с

class LightTimeSolver:
    def __init__(self):
        """
        Инициализация решателя light-time уравнений
        """
        self.c = c_km
        self.spice_epoch = datetime(1950, 1, 1, 0, 0, 0)
        self.spice_epoch_jd = 2433282.5  # JD для 1950-01-01 00:00:00
        
    def utc_to_tdb(self, time_utc):
        """
        Преобразует время из UTC в TDB (в секундах от эпохи SPICE)
        """
        try:
            # Конвертируем секунды в datetime
            dt_utc = self.spice_epoch + timedelta(seconds=float(time_utc))
            
            # Создаем объект Time в UTC
            t_utc = Time(dt_utc, scale='utc', format='datetime')
            
            # Преобразуем напрямую в TDB
            t_tdb = t_utc.tdb
            
            # Конвертируем обратно в секунды от SPICE эпохи
            time_tdb = (t_tdb.jd - self.spice_epoch_jd) * 86400.0
            
            return float(time_tdb)
            
        except Exception as e:
            print(f"Ошибка преобразования времени: {e}")
            # Упрощенное преобразование
            return float(time_utc) + 32.184
    
    def tdb_to_utc(self, time_tdb):
        """
        Преобразует время из TDB в UTC
        """
        try:
            # Конвертируем в JD
            jd_tdb = self.spice_epoch_jd + float(time_tdb) / 86400.0
            
            # Создаем объект Time в TDB
            t_tdb = Time(jd_tdb, format='jd', scale='tdb')
            
            # Преобразуем в UTC
            t_utc = t_tdb.utc
            
            # Конвертируем в секунды
            time_utc = (t_utc.jd - self.spice_epoch_jd) * 86400.0
            
            return float(time_utc)
        except Exception as e:
            print(f"Ошибка обратного преобразования: {e}")
            return float(time_tdb) - 32.184
    
    def compute_one_way_light_time(self, t3_utc, sc_pos_func, earth_pos_func, 
                                   max_iter=20, tol=1e-12):
        """
        Решает light-time уравнение для одностороннего измерения (КА -> Земля)
        """
        # Шаг 1: Преобразуем t3 из UTC в TDB
        t3_tdb = self.utc_to_tdb(t3_utc)
        
        # Шаг 2: Итеративное решение уравнения светового времени
        t2_guess = t3_tdb
        tau_prev = 0
        tau = 0
        
        for i in range(max_iter):
            # Получаем позиции
            r_sc = sc_pos_func(t2_guess)      # Позиция КА в момент передачи
            r_earth = earth_pos_func(t3_tdb)  # Позиция Земли в момент приема
            
            # Вычисляем расстояние
            rho = np.linalg.norm(r_sc - r_earth)
            
            # Время прохождения света (без поправок)
            tau = rho / self.c
            
            # Вычисляем релятивистские поправки
            delta_tau = self._compute_shapiro_correction(r_sc, r_earth)
            tau += delta_tau
            
            # Новая оценка t2
            t2_new = t3_tdb - tau
            
            # Проверяем сходимость
            delta = abs(tau - tau_prev)
            if delta < tol:
                t2_tdb = t2_new
                break
            
            if i == max_iter - 1:
                print(f"  Предупреждение: не достигнута сходимость за {max_iter} итераций")
                t2_tdb = t2_new
                break
            
            tau_prev = tau
            t2_guess = t2_new
        
        return t2_tdb, t3_tdb, tau, rho
    
    def _compute_shapiro_correction(self, r1, r2):
        """
        Вычисляет релятивистскую поправку Шапиро
        """
        r1_norm = np.linalg.norm(r1)
        r2_norm = np.linalg.norm(r2)
        r12 = np.linalg.norm(r2 - r1)
        
        # Гравитационная постоянная Солнца в км³/с²
        mu_sun = 1.3271244004193938e11
        
        # Параметр PPN γ (ОТО: γ = 1)
        gamma = 1.0
        
        if r12 > 0 and r1_norm > 0 and r2_norm > 0:
            term = (r1_norm + r2_norm + r12) / (r1_norm + r2_norm - r12)
            if term > 0:
                delta_tau = (2 * gamma * mu_sun / (c_km**3)) * np.log(term)
                return delta_tau
        
        return 0.0


def load_trajectory_data(simulation_file='direct_problem_results.npz'):
    """
    Загружает траектории из файла симуляции
    """
    print(f"Загрузка траекторий из {simulation_file}...")
    
    try:
        data = np.load(simulation_file, allow_pickle=True)
        
        # Извлекаем данные
        times_jd = data['times']  # Юлианские даты
        sc_positions = data['sc_positions']  # Позиции КА, км
        earth_positions = data['earth_positions']  # Позиции Земли, км
        
        print(f"  Загружено {len(times_jd)} точек")
        print(f"  Временной диапазон: JD {times_jd[0]:.2f} - {times_jd[-1]:.2f}")
        
        # Создаем интерполяционные функции
        # Конвертируем JD в секунды от SPICE эпохи
        spice_epoch_jd = 2433282.5
        times_seconds = (times_jd - spice_epoch_jd) * 86400.0
        
        # Функция для позиции КА
        sc_pos_interp = interp1d(times_seconds, sc_positions, axis=0, 
                                bounds_error=False, fill_value="extrapolate",
                                kind='linear')
        
        # Функция для позиции Земли
        earth_pos_interp = interp1d(times_seconds, earth_positions, axis=0,
                                   bounds_error=False, fill_value="extrapolate",
                                   kind='linear')
        
        return sc_pos_interp, earth_pos_interp, times_seconds[0], times_seconds[-1]
        
    except Exception as e:
        print(f"Ошибка загрузки траекторий: {e}")
        return None, None, None, None


def integrate_with_simulation(simulation_file='direct_problem_results.npz'):
    """
    Интегрирует light-time решатель с результатами симуляции
    """
    print(f"\nИнтеграция с симуляцией из файла: {simulation_file}")
    
    sc_pos_func, earth_pos_func, t_min, t_max = load_trajectory_data(simulation_file)
    
    if sc_pos_func is None:
        print("Ошибка: не удалось загрузить траектории")
        return None, None, None, None
    
    print(f"Диапазон доступного времени: {t_min:.1f} - {t_max:.1f} с от 1950-01-01")
    
    return sc_pos_func, earth_pos_func, t_min, t_max


def solve_light_time_for_odf(odf_csv_path='odf_data.csv', 
                            simulation_file='direct_problem_results.npz',
                            output_csv='light_time_results.csv'):
    """
    Решает light-time уравнения для всех записей в ODF файле
    """
    print("РЕШЕНИЕ LIGHT-TIME УРАВНЕНИЙ ДЛЯ ODF ДАННЫХ")
    
    # 1. Загружаем ODF данные
    print("\n1. Загрузка ODF данных...")
    try:
        df = pd.read_csv(odf_csv_path)
        
        # Конвертируем строку времени в datetime
        if 'record_time' in df.columns:
            df['record_time'] = pd.to_datetime(df['record_time'])
        
        print(f"   Загружено {len(df)} записей")
        
        # Преобразуем datetime в секунды от SPICE эпохи
        spice_epoch = datetime(1950, 1, 1, 0, 0, 0)
        df['t3_utc_seconds'] = df['record_time'].apply(
            lambda x: (x - spice_epoch).total_seconds() if pd.notnull(x) else 0
        )
        
    except Exception as e:
        print(f"   Ошибка загрузки ODF: {e}")
        return None
    
    # 2. Интегрируем с симуляцией
    sc_pos_func, earth_pos_func, t_min, t_max = integrate_with_simulation(simulation_file)
    
    if sc_pos_func is None:
        print("   Ошибка: не удалось загрузить траектории")
        return None
    
    # 3. Инициализируем решатель
    print("\n2. Инициализация решателя...")
    solver = LightTimeSolver()
    
    # 4. Решаем light-time уравнения для каждой записи
    print("\n3. Решение уравнений...")
    
    results = []
    failed_records = 0
    valid_records = 0
    
    for idx, row in df.iterrows():
        if idx % 100 == 0 and idx > 0:
            print(f"   Обработано {idx}/{len(df)} записей...")
        
        t3_utc = row['t3_utc_seconds']
        
        # Проверяем, попадает ли время в диапазон траекторий
        if t3_utc < t_min or t3_utc > t_max:
            failed_records += 1
            continue
        
        try:
            # Решаем light-time уравнение
            t2_tdb, t3_tdb, tau, distance = solver.compute_one_way_light_time(
                t3_utc,
                sc_pos_func,
                earth_pos_func,
                max_iter=15,
                tol=1e-12
            )
            
            # Преобразуем t2 обратно в UTC для удобства
            t2_utc = solver.tdb_to_utc(t2_tdb)
            
            # Сохраняем результаты
            results.append({
                'record_index': idx,
                't3_utc_seconds': t3_utc,
                't3_datetime': row['record_time'],
                't2_tdb_seconds': t2_tdb,
                't2_utc_seconds': t2_utc,
                't3_tdb_seconds': t3_tdb,
                'light_time_seconds': tau,
                'distance_km': distance,
                'observable': row.get('full_observable', 0),
                'station_id': row.get('receiving_station_id', 0)
            })
            
            valid_records += 1
            
        except Exception as e:
            failed_records += 1
            continue
    
    # 5. Сохраняем результаты
    print(f"\n4. Сохранение результатов...")
    print(f"   Успешно обработано: {valid_records} записей")
    print(f"   Не обработано: {failed_records} записей")
    
    if results:
        df_results = pd.DataFrame(results)
        
        # Сортируем по времени
        df_results = df_results.sort_values('t3_utc_seconds')
        
        # Сохраняем в CSV
        df_results.to_csv(output_csv, index=False)
        print(f"   Результаты сохранены в {output_csv}")
        
        # Статистика
        print("\n5. Статистика результатов:")
        print(f"   Среднее расстояние: {df_results['distance_km'].mean()/1e6:.3f} млн км")
        print(f"   Мин. расстояние: {df_results['distance_km'].min()/1e6:.3f} млн км")
        print(f"   Макс. расстояние: {df_results['distance_km'].max()/1e6:.3f} млн км")
        print(f"   Среднее время прохождения: {df_results['light_time_seconds'].mean():.3f} с")
        print(f"   Мин. время прохождения: {df_results['light_time_seconds'].min():.3f} с")
        print(f"   Макс. время прохождения: {df_results['light_time_seconds'].max():.3f} с")
        
        # Проверяем физическую корректность
        print("\n6. Проверка корректности:")
        
        # Вычисляем ожидаемое время прохождения света для среднего расстояния
        avg_distance = df_results['distance_km'].mean()
        expected_time = avg_distance / c_km
        
        print(f"   Среднее расстояние: {avg_distance/1e6:.3f} млн км")
        print(f"   Ожидаемое время света: {expected_time:.3f} с")
        print(f"   Среднее вычисленное время: {df_results['light_time_seconds'].mean():.3f} с")
        print(f"   Разница (релятивистская поправка): {df_results['light_time_seconds'].mean() - expected_time:.9f} с")
        
        # Первые 3 результата для проверки
        print(f"\n7. Первые 3 результата:")
        for i in range(min(3, len(df_results))):
            r = df_results.iloc[i]
            time_before_t3 = r['t3_utc_seconds'] - r['t2_utc_seconds']
            print(f"   Запись {int(r['record_index'])}:")
            print(f"     t3 (прием): {r['t3_datetime']}")
            print(f"     Расстояние: {r['distance_km']/1e6:.6f} млн км")
            print(f"     Время света: {r['light_time_seconds']:.6f} с")
            print(f"     t2 было {time_before_t3:.3f} с до t3 (должно быть ~{r['light_time_seconds']:.3f} с)")
        
        return df_results
    else:
        print("   Нет результатов для сохранения")
        return None


def analyze_light_time_results(results_csv='light_time_results.csv'):
    """
    Анализирует и визуализирует результаты light-time расчетов
    """
    print("АНАЛИЗ РЕЗУЛЬТАТОВ LIGHT-TIME")
    
    try:
        df = pd.read_csv(results_csv)
        
        # Конвертируем строку обратно в datetime
        if 't3_datetime' in df.columns:
            df['t3_datetime'] = pd.to_datetime(df['t3_datetime'])
        
        print(f"Загружено {len(df)} записей для анализа")
        
        # 1. Базовая статистика
        print("\n1. Базовая статистика:")
        print(f"   Временной диапазон: {df['t3_datetime'].min()} - {df['t3_datetime'].max()}")
        print(f"   Продолжительность: {(df['t3_datetime'].max() - df['t3_datetime'].min()).days} дней")
        
        
        
        # 3. Расчет релятивистских поправок
        print("\n2. Анализ релятивистских поправок:")
        
        # Вычисляем ожидаемое время по расстоянию (без поправок)
        expected_time_no_corr = df['distance_km'] / c_km
        
        # Разница (это и есть поправка)
        correction = df['light_time_seconds'] - expected_time_no_corr
        
        print(f"   Средняя релятивистская поправка: {correction.mean():.12f} с")
        print(f"   Мин. поправка: {correction.min():.12f} с")
        print(f"   Макс. поправка: {correction.max():.12f} с")
        print(f"   Стандартное отклонение: {correction.std():.12f} с")
        
        # Преобразуем в микросекунды для наглядности
        print(f"   Средняя поправка: {correction.mean() * 1e6:.3f} мкс")
        
        # 4. Проверка физической корректности
        print("\n3. Проверка физической корректности:")
        
        # Проверяем, что t2 всегда раньше t3
        time_diff = df['t3_utc_seconds'] - df['t2_utc_seconds']
        
        if (time_diff > 0).all():
            print(f"    Все t2 корректно предшествуют t3")
            print(f"   Средняя разница t3-t2: {time_diff.mean():.3f} с")
            print(f"   Теоретическая разница (световое время): {df['light_time_seconds'].mean():.3f} с")
            print(f"   Средняя ошибка: {abs(time_diff.mean() - df['light_time_seconds'].mean()):.6f} с")
        else:
            print(f"    Ошибка: некоторые t2 позже t3")
        
        # Проверяем согласованность расстояния и времени
        expected_from_time = time_diff * c_km
        distance_error = abs(df['distance_km'] - expected_from_time) / df['distance_km']
        
        print(f"   Средняя относительная ошибка расстояния: {distance_error.mean()*100:.6f}%")
        print(f"   Макс. относительная ошибка: {distance_error.max()*100:.6f}%")

    except Exception as e:
        print(f"Ошибка анализа: {e}")    


def run_full_processing(odf_csv="odf_data.csv", sim_results="direct_problem_results.npz"):
    """
    Запускает полную обработку данных light-time
    """
    print("="*70)
    print("СИСТЕМА РЕШЕНИЯ LIGHT-TIME УРАВНЕНИЙ")
    print("="*70)
    
    # Проверка наличия файлов
    print("\n1. Проверка файлов...")
    
    if not os.path.exists(odf_csv):
        print(f"    Файл {odf_csv} не найден!")
        print("   Запустите сначала parser.py для создания odf_data.csv")
        return None
    
    if not os.path.exists(sim_results):
        print(f"    Файл {sim_results} не найден!")
        print("   Запустите сначала main_simulation.py для создания direct_problem_results.npz")
        return None
    
    print(f"    Файл {odf_csv} найден")
    print(f"    Файл {sim_results} найден")
    
    # Запуск расчета
    print("\n2. Запуск расчета light-time уравнений...")
    results = solve_light_time_for_odf(
        odf_csv_path=odf_csv,
        simulation_file=sim_results,
        output_csv='light_time_results.csv'
    )
    
    if results is not None:
        # Анализ результатов
        print("\n3. Анализ результатов...")
        analyze_light_time_results('light_time_results.csv')
        
        print("\n" + "="*70)
        print("ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО")
        print("="*70)
        
        # Пример использования решателя для одиночного расчета
        print("\n4. Пример одиночного расчета:")
        
        # Загружаем траектории
        sc_pos_func, earth_pos_func, t_min, t_max = integrate_with_simulation(sim_results)
        solver = LightTimeSolver()
        
        # Выбираем среднее время из данных
        if len(results) > 0:
            avg_time = results['t3_utc_seconds'].mean()
            
            print(f"   Среднее время из данных: {avg_time:.1f} с от 1950-01-01")
            print(f"   Соответствует дате: {datetime(1950,1,1) + timedelta(seconds=avg_time)}")
            
            # Расчет для этого времени
            t2_tdb, t3_tdb, tau, distance = solver.compute_one_way_light_time(
                avg_time,
                sc_pos_func,
                earth_pos_func
            )
            
            print(f"\n   Результаты для среднего времени:")
            print(f"   Расстояние: {distance/1e6:.3f} млн км")
            print(f"   Время прохождения света: {tau:.6f} с")
            print(f"   Релятивистская поправка: {tau - distance/c_km:.12f} с")
            print(f"   Время передачи (t2): за {tau:.3f} с до приема")
    
    return results


def main():
    """
    Главная функция для запуска всей системы
    """
    # Параметры по умолчанию
    odf_csv = "odf_data.csv"
    sim_results = "direct_problem_results.npz"
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Использование: python light_time_done.py [опции]")
            print("Опции:")
            print("  -h, --help     Показать эту справку")
            print("  --odf FILE     Указать ODF CSV файл (по умолчанию: odf_data.csv)")
            print("  --sim FILE     Указать файл симуляции (по умолчанию: direct_problem_results.npz)")
            print("  --run          Запустить полную обработку (по умолчанию)")
            print("  --functions    Показать доступные функции")
            return
        
        if '--functions' in sys.argv:
            print("ДОСТУПНЫЕ ФУНКЦИИ:")
            print("1. solve_light_time_for_odf() - решение уравнений для ODF данных")
            print("2. analyze_light_time_results() - анализ результатов")
            print("3. integrate_with_simulation() - загрузка траекторий")
            print("4. LightTimeSolver() - класс для решения уравнений")
            print("5. run_full_processing() - полная обработка")
            print("\nПример использования:")
            print("  from light_time_done import run_full_processing")
            print("  results = run_full_processing('odf_data.csv', 'direct_problem_results.npz')")
            return
        
        # Обработка аргументов командной строки
        for i in range(1, len(sys.argv)):
            if sys.argv[i] == '--odf' and i+1 < len(sys.argv):
                odf_csv = sys.argv[i+1]
            elif sys.argv[i] == '--sim' and i+1 < len(sys.argv):
                sim_results = sys.argv[i+1]
    
    # Запуск полной обработки
    return run_full_processing(odf_csv, sim_results)


if __name__ == "__main__":
    main()