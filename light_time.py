# light_time.py
# Полная реализация light time solution по диссертации Ashok Kumar Verma (2013)

import numpy as np
from astropy.time import Time
import astropy.units as u

# Константы
C = 299792.458  # км/с
C_KM_PER_DAY = C * 86400.0  # км/сут
GAMMA = 1.0

# ============================================================
# КОНВЕРСИЯ UTC ↔ TDB
# ============================================================

def utc_to_tdb_jd(t3_utc_jd, t3_datetime):
    """Конвертация UTC в TDB."""
    t_utc = Time(t3_datetime, scale='utc', format='datetime')
    return t_utc.tdb.jd

def tdb_to_utc_jd(t_tdb_jd):
    """Конвертация TDB в UTC."""
    t_tdb = Time(t_tdb_jd, scale='tdb', format='jd')
    return t_tdb.utc.jd

# ============================================================
# ПОПРАВКА ШАПИРО ДЛЯ ОДНОЙ ВЕТВИ
# ============================================================

def shapiro_delay_one_way(r1, r2, bodies, t_mid_jd):
    """
    Вычисляет релятивистскую поправку Шапиро для одной ветви.
    
    Параметры:
    r1, r2: векторы положений (км)
    bodies: словарь с функциями положений тел и их GM
    t_mid_jd: момент времени в середине передачи (JD TDB)
    
    Возвращает:
    Поправку в секундах
    """
    delta = 0.0
    c3 = C ** 3  # км³/с³
    
    # Расстояние между станциями
    rho = np.linalg.norm(r2 - r1)
    
    # Поправка от Солнца
    if 'Sun' in bodies:
        gm_sun = bodies['Sun'][1]
        pos_func_sun = bodies['Sun'][0]
        r_s = pos_func_sun(t_mid_jd)
        
        rs1 = np.linalg.norm(r1 - r_s)
        rs2 = np.linalg.norm(r2 - r_s)
        
        denom = rs1 + rs2 - rho
        if denom > 1e-6:
            arg = (rs1 + rs2 + rho) / denom
            if arg > 1.0 + 1e-12:
                delta += (1 + GAMMA) * (gm_sun / c3) * np.log(arg)
    
    # Поправки от планет (кроме Солнца)
    for name, (pos_func, gm) in bodies.items():
        if name == 'Sun':
            continue
            
        r_b = pos_func(t_mid_jd)
        rb1 = np.linalg.norm(r1 - r_b)
        rb2 = np.linalg.norm(r2 - r_b)
        
        denom_b = rb1 + rb2 - rho
        if denom_b > 1e-6:
            arg_b = (rb1 + rb2 + rho) / denom_b
            if arg_b > 1.0 + 1e-12:
                delta += (1 + GAMMA) * (gm / c3) * np.log(arg_b)
    
    return delta

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def geometric_delay(r1, r2):
    """Геометрическая задержка между двумя точками в секундах."""
    return np.linalg.norm(r2 - r1) / C

# ============================================================
# DOWN-LEG (t2 → t3)
# ============================================================

def compute_down_leg(t3_tdb_jd, r3_func, r2_func, bodies, max_iter=50, tol=1e-12):
    """
    Вычисляет время передачи от КА к Земле (down-leg).
    
    Параметры:
    t3_tdb_jd: время приема на Земле (JD TDB)
    r3_func: функция положения Земли (приемник)
    r2_func: функция положения КА (передатчик)
    bodies: гравитирующие тела
    max_iter: максимальное число итераций
    tol: точность сходимости (дни)
    
    Возвращает:
    t2_tdb_jd: время передачи от КА (JD TDB)
    tau_d_days: время передачи в днях
    """
    # Начальное приближение: чисто геометрическая задержка
    r3 = r3_func(t3_tdb_jd)
    r2_guess = r2_func(t3_tdb_jd)
    tau_geom_days = np.linalg.norm(r3 - r2_guess) / C_KM_PER_DAY
    t2 = t3_tdb_jd - tau_geom_days
    
    for iter_num in range(max_iter):
        # Положения в текущем приближении
        r2 = r2_func(t2)
        r3 = r3_func(t3_tdb_jd)
        
        # Геометрическое расстояние
        rho = np.linalg.norm(r3 - r2)
        tau_geom_days = rho / C_KM_PER_DAY
        
        # Поправка Шапиро
        t_mid = (t2 + t3_tdb_jd) / 2.0
        delta_sec = shapiro_delay_one_way(r3, r2, bodies, t_mid)
        delta_days = delta_sec / 86400.0
        
        # Общая задержка
        tau_total_days = tau_geom_days + delta_days
        
        # Новое время передачи
        t2_new = t3_tdb_jd - tau_total_days
        
        # Проверка сходимости
        if abs(t2_new - t2) < tol:
            return t2_new, tau_total_days
        
        t2 = t2_new
    
    raise RuntimeError(f"Down-leg не сошёлся за {max_iter} итераций. "
                      f"Последняя разница: {abs(t2_new - t2):.2e} дней")

# ============================================================
# UP-LEG (t1 → t2)
# ============================================================

def compute_up_leg(t2_tdb_jd, r1_func, r2_func, bodies, max_iter=50, tol=1e-12):
    """
    Вычисляет время передачи от Земли к КА (up-leg).
    
    Параметры:
    t2_tdb_jd: время приема на КА (JD TDB)
    r1_func: функция положения Земли (передатчик)
    r2_func: функция положения КА (приемник)
    bodies: гравитирующие тела
    max_iter: максимальное число итераций
    tol: точность сходимости (дни)
    
    Возвращает:
    t1_tdb_jd: время передачи с Земли (JD TDB)
    tau_u_days: время передачи в днях
    """
    # Начальное приближение
    r2 = r2_func(t2_tdb_jd)
    r1_guess = r1_func(t2_tdb_jd)
    tau_geom_days = np.linalg.norm(r2 - r1_guess) / C_KM_PER_DAY
    tau_u_days = tau_geom_days
    
    for iter_num in range(max_iter):
        # Время передачи
        t1 = t2_tdb_jd - tau_u_days
        
        # Положения
        r1 = r1_func(t1)
        r2 = r2_func(t2_tdb_jd)
        
        # Геометрическое расстояние
        rho = np.linalg.norm(r2 - r1)
        tau_geom_days = rho / C_KM_PER_DAY
        
        # Поправка Шапиро
        t_mid = (t1 + t2_tdb_jd) / 2.0
        delta_sec = shapiro_delay_one_way(r1, r2, bodies, t_mid)
        delta_days = delta_sec / 86400.0
        
        # Общая задержка
        tau_new = tau_geom_days + delta_days
        
        # Проверка сходимости
        if abs(tau_new - tau_u_days) < tol:
            return t1, tau_new
        
        tau_u_days = tau_new
    
    raise RuntimeError(f"Up-leg не сошёлся за {max_iter} итераций. "
                      f"Последняя разница: {abs(tau_new - tau_u_days):.2e} дней")

# ============================================================
# ПОЛНОЕ LIGHT TIME РЕШЕНИЕ
# ============================================================

def compute_light_time(t3_utc_jd, t3_datetime, r1_func, r2_func, r3_func, bodies):
    """
    Вычисляет полное light time решение для двухстороннего измерения.
    
    Параметры:
    t3_utc_jd: время приема сигнала на Земле (JD UTC)
    t3_datetime: то же время как datetime объект
    r1_func: функция положения Земли-передатчика
    r2_func: функция положения КА
    r3_func: функция положения Земли-приемника
    bodies: словарь гравитирующих тел {'Name': (pos_func, gm)}
    
    Возвращает:
    Словарь с результатами
    """
    # Конвертация UTC → TDB
    t3_tdb_jd = utc_to_tdb_jd(t3_utc_jd, t3_datetime)
    
    # Вычисление down-leg (КА → Земля)
    t2_tdb_jd, tau_d_days = compute_down_leg(t3_tdb_jd, r3_func, r2_func, bodies)
    
    # Вычисление up-leg (Земля → КА)
    t1_tdb_jd, tau_u_days = compute_up_leg(t2_tdb_jd, r1_func, r2_func, bodies)
    
    # Конвертация обратно в UTC для вывода
    t1_utc_jd = tdb_to_utc_jd(t1_tdb_jd)
    t2_utc_jd = tdb_to_utc_jd(t2_tdb_jd)
    
    # Времена в секундах
    total_delay_sec = (tau_u_days + tau_d_days) * 86400.0
    tau_up_sec = tau_u_days * 86400.0
    tau_down_sec = tau_d_days * 86400.0
    
    # Чисто геометрические расстояния
    r1 = r1_func(t1_tdb_jd)
    r2 = r2_func(t2_tdb_jd)
    r3 = r3_func(t3_tdb_jd)
    
    geom_up = np.linalg.norm(r2 - r1) / C
    geom_down = np.linalg.norm(r3 - r2) / C
    geometric_sec = geom_up + geom_down
    
    # Поправка Шапиро
    shapiro_sec = total_delay_sec - geometric_sec
    
    # Расстояния
    distance_up = np.linalg.norm(r2 - r1)  # км
    distance_down = np.linalg.norm(r3 - r2)  # км
    total_distance = distance_up + distance_down
    
    return {
        # Времена в TDB
        't1_tdb_jd': t1_tdb_jd,
        't2_tdb_jd': t2_tdb_jd,
        't3_tdb_jd': t3_tdb_jd,
        
        # Времена в UTC
        't1_utc_jd': t1_utc_jd,
        't2_utc_jd': t2_utc_jd,
        't3_utc_jd': t3_utc_jd,
        
        # Задержки
        'total_delay_sec': total_delay_sec,
        'tau_up_sec': tau_up_sec,
        'tau_down_sec': tau_down_sec,
        'tau_up_days': tau_u_days,
        'tau_down_days': tau_d_days,
        
        # Компоненты
        'geometric_sec': geometric_sec,
        'shapiro_sec': shapiro_sec,
        
        # Расстояния
        'distance_up_km': distance_up,
        'distance_down_km': distance_down,
        'total_distance_km': total_distance,
        
        # Геометрические расстояния в а.е.
        'distance_up_au': distance_up / 149597870.7,
        'distance_down_au': distance_down / 149597870.7,
        
        # Временные интервалы
        't2_minus_t1_days': t2_utc_jd - t1_utc_jd,
        't3_minus_t2_days': t3_utc_jd - t2_utc_jd,
        't3_minus_t1_days': t3_utc_jd - t1_utc_jd
    }