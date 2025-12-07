# parse_range_final.py
# Работает в той же папке, где лежат .xml и .dat файлы
# Выдаёт только дальности (type 36, 37, 41) в метрах

import numpy as np
import pandas as pd
from pds4_tools import pds4_read
from datetime import datetime, timedelta
import glob

EPOCH_1950 = datetime(1950, 1, 1)
C = 299792458.0

print("Ищу ODF-файлы в текущей папке...")
xml_files = sorted(glob.glob("*.xml"))

if not xml_files:
    print("ОШИБКА: нет .xml файлов в этой папке!")
    exit()

print(f"Найдено {len(xml_files)} файлов\n")

all_dfs = []

for xml_path in xml_files:
    print(f"Обрабатываю: {xml_path}")
    
    try:
        structures = pds4_read(xml_path, quiet=True)
        table = structures['ODF Orbit Data Group Data']
        data = table.data
    except Exception as e:
        print(f"   → Ошибка чтения: {e}")
        continue

    # === Поля ===
    time_int   = data['Record Time Tag, integer part'].astype(np.float64)
    items_2_3  = data['Items 2-3']
    obs_int    = data['Observable, integer part']
    obs_frac   = data['Observable, fractional part']
    items_6_14 = data['Items 6-14']

    # === Функция извлечения битов (надёжная) ===
    def extract_bits(byte_array, start_bit_1, length):
        vals = []
        for b in byte_array:
            if len(b) == 0:
                vals.append(0)
                continue
            v = int.from_bytes(b, 'big')
            shift = 32 - (start_bit_1 + length - 1)
            vals.append((v >> shift) & ((1 << length) - 1))
        return np.array(vals)

    format_id    = extract_bits(items_6_14, 1,  3)   # биты 1–3
    data_type    = extract_bits(items_6_14, 20, 6)  # биты 20–25
    validity     = extract_bits(items_6_14, 32, 1)  # бит 32

    # Дробная часть времени (мс) — биты 1–10 из Items 2-3
    frac_ms = extract_bits(items_2_3, 1, 10)
    time_sec = time_int + frac_ms / 1000.0

    # Наблюдаемая величина в наносекундах
    observable_ns = obs_int + obs_frac * 1e-9

    # === ФИЛЬТР: только дальности и валидные записи ===
    mask = (format_id == 2) & \
           (np.isin(data_type, [36, 37, 41])) & \
           (validity == 0)

    n = mask.sum()
    if n == 0:
        print(f"   → дальностей нет в этом файле")
        continue

    range_m = observable_ns[mask] * 1e-9 * C / 2.0

    print(f"   НАЙДЕНО {n} измерений дальности!")

    df = pd.DataFrame({
        'time_sec_1950': time_sec[mask],
        'time_utc':      [EPOCH_1950 + timedelta(seconds=t) for t in time_sec[mask]],
        'range_meters':  range_m,
        'data_type':     data_type[mask],
    })
    all_dfs.append(df)

# === ИТОГ ===
if all_dfs:
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values('time_sec_1950').reset_index(drop=True)

    print("\n" + "="*60)
    print(f"УСПЕХ! Всего измерений дальности: {len(result)}")
    print(f"Период: {result['time_utc'].iloc[0]} → {result['time_utc'].iloc[-1]}")
    print(f"Дальность: {result['range_meters'].min()/1e6:.2f} – {result['range_meters'].max()/1e6:.2f} млн км")

    # Сохраняем
    result.to_csv("messenger_range_all.csv", index=False)
    np.savez("messenger_range_all.npz",
             time=np.array(result['time_sec_1950']),
             range=np.array(result['range_meters']))

    print("\nГотово! Файлы сохранены:")
    print("   messenger_range_all.csv")
    print("   messenger_range_all.npz  ← используй этот для МНК")
else:
    print("\nВ этих 11 файлах дальностей нет (только доплер).")
    print("Это нормально для января 2015 — попробуй файлы за март–апрель 2015,")
    print("там точно есть type 41 (RE Range).")
