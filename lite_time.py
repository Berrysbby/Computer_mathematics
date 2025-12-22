# light_time.py

import numpy as np
from datetime import datetime, timedelta

# Константы
c = 299792458.0  # скорость света, м/с
SPICE_EPOCH = datetime(1950, 1, 1, 0, 0, 0)

def utc_to_tdb(time_utc, time_datetime):
    # Константы из Verma
    Lg = 6.969290134e-10  # Отношение TCG-TT (безразмерное)
    T_0 = 2443144.5003725  # JD на 1977-01-01 00:00:00 TAI
    Lb = 1.550519768e-8   # Отношение TCB-TDB (безразмерное)
    TDB_0 = -6.55e5       # Смещение TDB_0 в секундах
    
    # 1. UTC - TAI (високосные секунды)
    t = Time(time_datetime, scale='utc')
    delta_tai = erfa.dat(t.jd1, t.jd2)  # TAI - UTC в секундах
    time_tai = time_utc + delta_tai
    
    # 2. TAI - TT (постоянное смещение)
    TT_minus_TAI = 32.184  # секунд
    time_tt = time_tai + TT_minus_TAI
    
    # 3. TT - TDB (упрощённо через astropy)
    spice_epoch_jd = 2433282.5  
    
    # Преобразуем секунды SPICE в JD TT
    jd_tt = spice_epoch_jd + time_tt / 86400.0
    
    # Создаём Time объект в шкале TT
    t_tt = Time(jd_tt, format='jd', scale='tt')
    
    # Преобразуем в TDB
    t_tdb = t_tt.tdb
    
    # Возвращаем в секундах SPICE
    jd_tdb = t_tdb.jd
    time_tdb = (jd_tdb - spice_epoch_jd) * 86400.0
    
    return time_tdb

