# light_time_unified.py

import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import pandas as pd
import os

# Константы
C_KM = 299792.458  # скорость света, км/с
SPICE_EPOCH_JD = 2433282.5  # J1950
SPICE_EPOCH = datetime(1950, 1, 1, 0, 0, 0)

def _jd_to_sec(jd):
    """Конвертирует время из JD в секунды TDB от J1950"""
    return (jd - SPICE_EPOCH_JD) * 86400.0


def load_jpl_vector_table(filepath):
    """
    Загружает векторную таблицу JPL Horizons
    Возвращает функцию r(t) в SSB
    """
    times_jd = []
    positions = []

    with open(filepath, 'r') as f:
        in_block = False
        for line in f:
            line = line.strip()

            if '$$SOE' in line:
                in_block = True
                continue
            if '$$EOE' in line:
                break

            if in_block and line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    try:
                        jd = float(parts[0])
                        x = float(parts[2])
                        y = float(parts[3])
                        z = float(parts[4])
                        times_jd.append(jd)
                        positions.append([x, y, z])
                    except ValueError:
                        pass

    times_sec = _jd_to_sec(np.array(times_jd))
    positions = np.array(positions)

    pos_func = interp1d(
        times_sec,
        positions,
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
        kind="cubic"
    )

    return pos_func


def load_stations_cmt(filepath):
    stations = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or 'Antenna' in line or 'Diameter' in line or 'x (m)' in line or '---' in line:
                continue
            
            parts = line.split()
            
            if len(parts) < 5:
                continue
            
            if parts[0].upper() != 'DSS':
                continue
            
            try:
                station_num = parts[1]
                station_num = ''.join(c for c in station_num if c.isdigit())
                
                if not station_num:
                    continue
                
                coords_found = []
                for part in parts[2:]:
                    clean_part = part.replace('{', '').replace('}', '').replace('+', '')
                    try:
                        coord = float(clean_part)
                        coords_found.append(coord)
                        if len(coords_found) == 3:
                            break
                    except ValueError:
                        continue
                
                if len(coords_found) == 3:
                    x, y, z = coords_found
                    x, y, z = x/1000.0, y/1000.0, z/1000.0
                    
                    station_id_num = station_num
                    station_id_full = f'DSS-{station_num}'
                    
                    stations[station_id_num] = np.array([x, y, z])
                    stations[station_id_full] = np.array([x, y, z])
                    
            except (ValueError, IndexError) as e:
                continue
    
    print(f"Загружено {len(stations)//2} уникальных станций")
    return stations


def _get_station_vector_inertial(station_id, t_sec):
    """
    Возвращает вектор станции в инерциальной системе (с учетом вращения Земли)
    """
    
    # Получаем станцию по ID 
    station_key = str(station_id)
    r_e_3 = _stations[station_key].copy()

    omega = 7.2921151467e-5  # рад/с
    angle = omega * t_sec

    c = np.cos(angle)
    s = np.sin(angle)

    x = r_e_3[0] * c - r_e_3[1] * s
    y = r_e_3[0] * s + r_e_3[1] * c

    r_e_3[0] = x
    r_e_3[1] = y

    return r_e_3


def initialize_data(data_dir="jpl_data"):
    """
    Инициализирует все необходимые данные для работы функций
    """
    global _earth_pos, _mercury_pos, _messenger_pos, _stations
    
    _earth_pos = load_jpl_vector_table(os.path.join(data_dir, "earth.txt"))
    _mercury_pos = load_jpl_vector_table(os.path.join(data_dir, "mercury.txt"))
    _messenger_pos = load_jpl_vector_table(os.path.join(data_dir, "messenger.txt"))
    
    print("Загрузка данных JPL Horizons завершена.")
    _stations = load_stations_cmt(os.path.join(data_dir, "earthstns_fx_201023.cmt.txt"))
    print("Загрузка данных станций завершена.")


def get_station_vector(station_id, t_sec):
    
    # Земля относительно SSB
    r_c_e = _earth_pos(t_sec)
    
    # Станция относительно Земли (с учетом вращения)
    r_e_3 = _get_station_vector_inertial(station_id, t_sec)
    
    # Станция относительно SSB
    r_c_3 = r_c_e + r_e_3
    
    return r_c_3


def get_spacecraft_vector(t_sec):
    """
    Возвращает вектор космического аппарата (MESSENGER) 
    относительно барицентра Солнечной системы (SSB)
    """
    
    # MESSENGER относительно SSB
    r_c_2 = _messenger_pos(t_sec)
    
    return r_c_2


class LightTimeSolver:
    def __init__(self):
        """
        Инициализация решателя light-time уравнений
        """
        self.c = C_KM
        
        # Гравитационные параметры (км³/с²)
        self.mu_sun = 1.3271244004193938e11
        """
        self.mu_planets = {
            'mercury': 2.20329e4,
            'venus': 3.24859e5,
            'earth': 3.986004415e5,
            'mars': 4.2828372e4,
            'jupiter': 1.26712767863e8,
            'saturn': 3.7940626063e7,
            'uranus': 5.794549007e6,
            'neptune': 6.836534064e6,
            'pluto': 9.816e2,
            'moon': 4.902800066e3
        }
        """

    def utc_to_tdb(self, t_utc_seconds):
        leap_seconds = 36  
        t_tdb = t_utc_seconds + 32.184 + leap_seconds
        return t_tdb

    def tdb_to_utc(self, t_tdb_seconds):
        leap_seconds = 36
        t_utc = t_tdb_seconds - 32.184 - leap_seconds
        return t_utc

    def _compute_shapiro_correction(self, r1, r2):
        r1_norm = np.linalg.norm(r1)
        r2_norm = np.linalg.norm(r2)
        r12 = np.linalg.norm(r2 - r1)
        
        if r12 > 0 and r1_norm > 0 and r2_norm > 0:
            numerator = r1_norm + r2_norm + r12
            denominator = r1_norm + r2_norm - r12
            
            if denominator > 0:
                delta_tau = (2 * self.mu_sun / (self.c**3)) * np.log(numerator / denominator)
                return delta_tau
        
        return 0.0

    def compute_one_way_light_time(self, t3_utc, station_id, max_iter=20, tol=1e-12):
        t3_tdb = self.utc_to_tdb(t3_utc)

        t2_guess = t3_tdb
        
        for i in range(max_iter):
            r3_C = get_station_vector(station_id, t3_tdb) 
            r2_C = get_spacecraft_vector(t2_guess) 
            
            rho = np.linalg.norm(r2_C - r3_C)
            tau = rho / self.c
            
            delta_tau = self._compute_shapiro_correction(r2_C, r3_C)
            tau += delta_tau
       
            t2_guess = t3_tdb - tau
     
        return t2_guess, t3_tdb, tau, rho

    def compute_two_way_light_time(self, t3_utc, station_id, max_iter=20, tol=1e-12):
        t2_tdb, t3_tdb, tau_down, _ = self.compute_one_way_light_time(
            t3_utc, station_id, max_iter, tol
        )
        
        t1_guess = t2_tdb
        r2_C = get_spacecraft_vector(t2_tdb)
        for i in range(max_iter):
            r1_C = get_station_vector(station_id, t1_guess)


            rho_up = np.linalg.norm(r2_C - r1_C)
            tau_up = rho_up / self.c
            
            delta_tau = self._compute_shapiro_correction(r1_C, r2_C)
            tau_up += delta_tau
            
            t1_guess = t2_tdb - tau_up
        
        
        return t1_guess, t2_tdb, t3_tdb, tau_up, tau_down


def extract_station_id_from_odf(row):
    station_str = str(row['receiving_station_id'])
    if station_str.startswith('DSS-'):
        return station_str
    elif station_str.isdigit():
        return f"DSS-{station_str}"


def solve_light_time_for_odf(odf_csv_path='odf_data.csv', output_csv='light_time_results.csv'):

    initialize_data("jpl_data")
    
    solver = LightTimeSolver()
    
    try:
        df = pd.read_csv(odf_csv_path)
        
        if 'record_time' in df.columns:
            df['record_time'] = pd.to_datetime(df['record_time'])
            
            def datetime_to_seconds(dt):
                if pd.isna(dt):
                    return 0
                return (dt - SPICE_EPOCH).total_seconds()
            
            df['t3_utc_seconds'] = df['record_time'].apply(datetime_to_seconds)
        else:
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_cols:
                df['record_time'] = pd.to_datetime(df[time_cols[0]])
                def datetime_to_seconds(dt):
                    if pd.isna(dt):
                        return 0
                    return (dt - SPICE_EPOCH).total_seconds()
                df['t3_utc_seconds'] = df['record_time'].apply(datetime_to_seconds)
            else:
                print("Ошибка: не найдена колонка с временем")
                return None
                
    except Exception as e:
        print(f"Ошибка загрузки ODF: {e}")
        return None
    
    df['station_id'] = df.apply(extract_station_id_from_odf, axis=1)
    
    results = []
    total_records = len(df)
    processed_records = 0
    
    for idx, row in df.iterrows():
        t3_utc = row['t3_utc_seconds']
        station_id = row['station_id']
        
        try:
            t1_tdb, t2_tdb, t3_tdb, tau_up, tau_down = solver.compute_two_way_light_time(
                t3_utc, station_id, max_iter=15, tol=1e-12
            )
            
            t1_utc = solver.tdb_to_utc(t1_tdb)
            t2_utc = solver.tdb_to_utc(t2_tdb)
            
            results.append({
                'record_index': idx,
                'station_id': station_id,
                't3_utc_seconds': t3_utc,
                't1_tdb_seconds': t1_tdb,
                't1_utc_seconds': t1_utc,
                't2_tdb_seconds': t2_tdb,
                't2_utc_seconds': t2_utc,
                't3_tdb_seconds': t3_tdb,
                'tau_up_seconds': tau_up,
                'tau_down_seconds': tau_down,
                'total_light_time_seconds': tau_up + tau_down,
                'range_uplink_km': tau_up * C_KM,
                'range_downlink_km': tau_down * C_KM
            })
            
            processed_records += 1
            
        except Exception as e:
            print(f"Ошибка для записи {idx} (станция {station_id}, время {t3_utc}): {e}")
            continue
    
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('t3_utc_seconds')
        df_results.to_csv(output_csv, index=False)
        
        print(f"Всего записей в ODF: {total_records}")
        print(f"Успешно обработано: {processed_records}")
        print(f"Не удалось обработать: {total_records - processed_records}")
        print(f"Результаты сохранены в: {output_csv}")
        
        if len(df_results) > 0:
            print(f"\nПРИМЕР ПЕРВОЙ ЗАПИСИ:")
            first = df_results.iloc[0]
            print(f"Станция: {first['station_id']}")
            print(f"Uplink время: {first['tau_up_seconds']:.6f} с")
            print(f"Downlink время: {first['tau_down_seconds']:.6f} с")
            print(f"Общее время: {first['total_light_time_seconds']:.6f} с")
            print(f"Расстояние uplink: {first['range_uplink_km']:.3f} км")
            print(f"Расстояние downlink: {first['range_downlink_km']:.3f} км")
    
        return df_results
    else:
        print("Нет результатов для сохранения")
        return None


def main():
    
    odf_csv = "odf_data.csv"
    output_csv = 'light_time_results.csv'
    
    if not os.path.exists(odf_csv):
        print(f"\nФайл {odf_csv} не найден!")
    
    print(f"ODF файл: {odf_csv}")
    print(f"Выходной файл: {output_csv}")

    results = solve_light_time_for_odf(
        odf_csv_path=odf_csv,
        output_csv=output_csv
    )
    
    if results is not None:
        print(f"\nРасчеты успешно завершены.")
    else:
        print(f"\nНе удалось выполнить расчеты.")

if __name__ == "__main__":
    main()