# test_light_time.py
# Полный тест light time решения

import os
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from astropy.time import Time

from main_simulation import load_jpl_vector_table, GM_SUN, GM_MERCURY, GM_EARTH
from light_time import compute_light_time

print("=" * 80)
print("ТЕСТ LIGHT TIME РЕШЕНИЯ")
print("=" * 80)

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ СИМУЛЯЦИИ
# ============================================================================

print("\n1. ЗАГРУЗКА СИМУЛЯЦИИ")
print("-" * 40)

try:
    data = np.load("direct_problem_results.npz")
    times = data['times']
    sc_positions = data['sc_positions']
    earth_positions = data['earth_positions']
    
    print(f"Загружено {len(times)} точек")
    print(f"Диапазон JD: {times.min():.5f} - {times.max():.5f}")
    
    # Проверка размерностей
    print(f"Размер sc_positions: {sc_positions.shape}")
    print(f"Размер earth_positions: {earth_positions.shape}")
    
    # Создание интерполяционных функций
    r_sc_func = interp1d(times, sc_positions, axis=0, fill_value="extrapolate")
    r_earth_func = interp1d(times, earth_positions, axis=0, fill_value="extrapolate")
    
    print("✓ Интерполяционные функции созданы")
    
except Exception as e:
    print(f"✗ Ошибка загрузки симуляции: {e}")
    exit()

# ============================================================================
# 2. ЗАГРУЗКА ЭФЕМЕРИД JPL
# ============================================================================

print("\n2. ЗАГРУЗКА ЭФЕМЕРИД JPL")
print("-" * 40)

base_dir = "jpl_data"
bodies = {}

# Список тел для загрузки
body_list = [
    ("Sun", GM_SUN, "sun.txt"),
    ("Mercury", GM_MERCURY, "mercury.txt"),
    ("Earth", GM_EARTH, "earth.txt")
]

for name, gm, fname in body_list:
    try:
        filepath = os.path.join(base_dir, fname)
        print(f"Загрузка {filepath}...")
        
        pos_func, vel_func, times_jpl, positions_jpl = load_jpl_vector_table(filepath, name)
        bodies[name] = (pos_func, gm)
        
        print(f"  ✓ Загружено {len(times_jpl)} точек")
        print(f"  Диапазон JD: {times_jpl[0]:.2f} – {times_jpl[-1]:.2f}")
        
    except Exception as e:
        print(f"  ✗ Ошибка загрузки {name}: {e}")

print(f"\nЗагружено тел: {list(bodies.keys())}")

# ============================================================================
# 3. ЗАГРУЗКА ODF-ДАННЫХ
# ============================================================================

print("\n3. ЗАГРУЗКА ODF-ДАННЫХ")
print("-" * 40)

try:
    odf_data = np.load("odf_data.npz")
    odf_time_sec = odf_data['time']
    
    # Конвертация секунд в JD UTC
    JD_1950 = 2433282.5
    odf_jd_utc = odf_time_sec / 86400.0 + JD_1950
    
    print(f"Загружено {len(odf_jd_utc)} точек ODF")
    print(f"Диапазон ODF JD: {odf_jd_utc.min():.5f} - {odf_jd_utc.max():.5f}")
    
    # Выбор точек в диапазоне симуляции
    mask = (odf_jd_utc >= times.min()) & (odf_jd_utc <= times.max())
    selected_jd = odf_jd_utc[mask]
    
    print(f"Выбрано {len(selected_jd)} точек в диапазоне симуляции")
    
    if len(selected_jd) == 0:
        print("✗ Нет точек ODF в диапазоне симуляции!")
        exit()
        
except Exception as e:
    print(f"✗ Ошибка загрузки ODF: {e}")
    exit()

# ============================================================================
# 4. РАСЧЕТ LIGHT TIME
# ============================================================================

print("\n4. РАСЧЕТ LIGHT TIME")
print("=" * 80)

# Настройки теста
n_test_points = min(20, len(selected_jd))  # Количество тестовых точек
results = []

print("\nРЕЗУЛЬТАТЫ РАСЧЕТА:")
print("-" * 120)
header = f"{'№':>3} | {'t3 (UTC)':>12} | {'t2 (UTC)':>12} | {'t1 (UTC)':>12} | "
header += f"{'Δt₁→₂':>8} | {'Δt₂→₃':>8} | {'Total':>8} | {'Shapiro':>10} | "
header += f"{'d₁ (а.е.)':>10} | {'d₂ (а.е.)':>10}"
print(header)
print("-" * 120)

for i, t3_utc_jd in enumerate(selected_jd[:n_test_points]):
    # Конвертация JD UTC в datetime
    unix_time = (t3_utc_jd - 2440587.5) * 86400.0
    t3_datetime = datetime.fromtimestamp(unix_time, tz=timezone.utc)
    
    try:
        # Вычисление light time
        res = compute_light_time(
            t3_utc_jd=t3_utc_jd,
            t3_datetime=t3_datetime,
            r1_func=r_earth_func,
            r2_func=r_sc_func,
            r3_func=r_earth_func,
            bodies=bodies
        )
        
        # Форматированный вывод
        print(f"{i+1:3d} | "
              f"{res['t3_utc_jd']:12.6f} | "
              f"{res['t2_utc_jd']:12.6f} | "
              f"{res['t1_utc_jd']:12.6f} | "
              f"{res['tau_up_sec']:8.3f} | "
              f"{res['tau_down_sec']:8.3f} | "
              f"{res['total_delay_sec']:8.3f} | "
              f"{res['shapiro_sec']*1e6:10.2f} | "
              f"{res['distance_up_au']:10.6f} | "
              f"{res['distance_down_au']:10.6f}")
        
        results.append(res)
        
    except Exception as e:
        print(f"{i+1:3d} | ОШИБКА: {e}")
        continue

print("-" * 120)

# ============================================================================
# 5. СТАТИСТИКА И АНАЛИЗ
# ============================================================================

if results:
    print("\n5. СТАТИСТИЧЕСКИЙ АНАЛИЗ")
    print("-" * 40)
    
    # Извлечение данных
    tau_up = np.array([r['tau_up_sec'] for r in results])
    tau_down = np.array([r['tau_down_sec'] for r in results])
    total_times = np.array([r['total_delay_sec'] for r in results])
    shapiro_vals = np.array([r['shapiro_sec'] for r in results])
    distances_up = np.array([r['distance_up_km'] for r in results])
    distances_down = np.array([r['distance_down_km'] for r in results])
    
    print("Средние значения:")
    print(f"  Время up-leg:      {np.mean(tau_up):.3f} ± {np.std(tau_up):.3f} с")
    print(f"  Время down-leg:    {np.mean(tau_down):.3f} ± {np.std(tau_down):.3f} с")
    print(f"  Общее время:       {np.mean(total_times):.3f} ± {np.std(total_times):.3f} с")
    print(f"  Поправка Шапиро:   {np.mean(shapiro_vals)*1e6:.2f} ± {np.std(shapiro_vals)*1e6:.2f} мкс")
    print(f"  Расстояние up:     {np.mean(distances_up)/1e6:.3f} ± {np.std(distances_up)/1e6:.3f} млн км")
    print(f"  Расстояние down:   {np.mean(distances_down)/1e6:.3f} ± {np.std(distances_down)/1e6:.3f} млн км")
    print(f"  Суммарное расстояние: {np.mean(distances_up + distances_down)/1e6:.3f} млн км")
    
    # Отношение времен up/down
    ratio = tau_up / tau_down
    print(f"  Отношение up/down:  {np.mean(ratio):.4f} ± {np.std(ratio):.4f}")
    
    # Проверка симметрии
    symmetry_diff = np.abs(tau_up - tau_down)
    print(f"  Разница up-down:   {np.mean(symmetry_diff):.3f} с (макс: {np.max(symmetry_diff):.3f} с)")
    
    # ============================================================================
    # 6. ГРАФИКИ
    # ============================================================================
    
    print("\n6. ПОСТРОЕНИЕ ГРАФИКОВ")
    print("-" * 40)
    
    t3_times = [r['t3_utc_jd'] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # График 1: Поправка Шапиро
    ax = axes[0, 0]
    ax.plot(t3_times, shapiro_vals * 1e6, 'bo-', linewidth=2, markersize=5)
    ax.set_xlabel('Время приема t3 (JD UTC)')
    ax.set_ylabel('Поправка Шапиро (мкс)')
    ax.set_title('Релятивистская поправка Шапиро')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=np.mean(shapiro_vals)*1e6, color='r', linestyle='--', alpha=0.5, 
               label=f'Среднее: {np.mean(shapiro_vals)*1e6:.1f} мкс')
    ax.legend()
    
    # График 2: Времена передачи
    ax = axes[0, 1]
    ax.plot(t3_times, tau_up, 'go-', linewidth=2, markersize=5, label='Up-leg (t1→t2)')
    ax.plot(t3_times, tau_down, 'ro-', linewidth=2, markersize=5, label='Down-leg (t2→t3)')
    ax.set_xlabel('Время приема t3 (JD UTC)')
    ax.set_ylabel('Время передачи (с)')
    ax.set_title('Время передачи по ветвям')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # График 3: Общее время
    ax = axes[0, 2]
    ax.plot(t3_times, total_times, 'mo-', linewidth=2, markersize=5)
    ax.set_xlabel('Время приема t3 (JD UTC)')
    ax.set_ylabel('Общее время (с)')
    ax.set_title('Общее время прохождения сигнала')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=np.mean(total_times), color='r', linestyle='--', alpha=0.5,
               label=f'Среднее: {np.mean(total_times):.1f} с')
    ax.legend()
    
    # График 4: Расстояния
    ax = axes[1, 0]
    ax.plot(t3_times, distances_up / 1e6, 'go-', linewidth=2, markersize=5, label='Земля→КА')
    ax.plot(t3_times, distances_down / 1e6, 'ro-', linewidth=2, markersize=5, label='КА→Земля')
    ax.set_xlabel('Время приема t3 (JD UTC)')
    ax.set_ylabel('Расстояние (млн км)')
    ax.set_title('Геометрические расстояния')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # График 5: Отношение времен
    ax = axes[1, 1]
    ax.plot(t3_times, ratio, 'co-', linewidth=2, markersize=5)
    ax.set_xlabel('Время приема t3 (JD UTC)')
    ax.set_ylabel('Отношение τ_up / τ_down')
    ax.set_title('Симметрия времен передачи')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Идеальная симметрия')
    ax.axhline(y=np.mean(ratio), color='g', linestyle='--', alpha=0.5,
               label=f'Среднее: {np.mean(ratio):.3f}')
    ax.legend()
    
    # График 6: Временные интервалы
    ax = axes[1, 2]
    t2_minus_t1 = [r['t2_utc_jd'] - r['t1_utc_jd'] for r in results]
    t3_minus_t2 = [r['t3_utc_jd'] - r['t2_utc_jd'] for r in results]
    
    ax.plot(t3_times, np.array(t2_minus_t1) * 86400, 'bo-', linewidth=2, markersize=5, label='t2 - t1')
    ax.plot(t3_times, np.array(t3_minus_t2) * 86400, 'ro-', linewidth=2, markersize=5, label='t3 - t2')
    ax.set_xlabel('Время приема t3 (JD UTC)')
    ax.set_ylabel('Интервал (с)')
    ax.set_title('Временные интервалы между событиями')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle(f'Light Time Solution для MESSENGER (n={len(results)} точек)', fontsize=16)
    plt.tight_layout()
    
    # Сохранение графиков
    plt.savefig('light_time_analysis_full.png', dpi=150, bbox_inches='tight')
    print("✓ Графики сохранены в light_time_analysis_full.png")
    
    plt.show()
    
    # ============================================================================
    # 7. ПРИМЕР ПОДРОБНОГО ВЫВОДА ДЛЯ ПЕРВОЙ ТОЧКИ
    # ============================================================================
    
    print("\n7. ПОДРОБНЫЙ АНАЛИЗ ПЕРВОЙ ТОЧКИ")
    print("-" * 40)
    
    if results:
        r = results[0]
        print(f"Время приема t3:      {r['t3_utc_jd']:.8f} JD UTC")
        print(f"                     {Time(r['t3_utc_jd'], format='jd', scale='utc').iso}")
        print()
        print(f"Время на КА t2:       {r['t2_utc_jd']:.8f} JD UTC")
        print(f"                     {Time(r['t2_utc_jd'], format='jd', scale='utc').iso}")
        print(f"  Δt(t3-t2):         {r['tau_down_sec']:.6f} с")
        print(f"  Расстояние:        {r['distance_down_km']/1e6:.6f} млн км")
        print()
        print(f"Время отправки t1:    {r['t1_utc_jd']:.8f} JD UTC")
        print(f"                     {Time(r['t1_utc_jd'], format='jd', scale='utc').iso}")
        print(f"  Δt(t2-t1):         {r['tau_up_sec']:.6f} с")
        print(f"  Расстояние:        {r['distance_up_km']/1e6:.6f} млн км")
        print()
        print(f"ОБЩИЕ РЕЗУЛЬТАТЫ:")
        print(f"  Суммарное время:   {r['total_delay_sec']:.6f} с")
        print(f"  Геометрическое:    {r['geometric_sec']:.6f} с")
        print(f"  Поправка Шапиро:   {r['shapiro_sec']:.9f} с")
        print(f"                     {r['shapiro_sec']*1e6:.3f} мкс")
        print()
        print(f"РАССТОЯНИЯ:")
        print(f"  Земля→КА:          {r['distance_up_au']:.6f} а.е.")
        print(f"  КА→Земля:          {r['distance_down_au']:.6f} а.е.")
        print(f"  Суммарное:         {r['total_distance_km']/1e6:.6f} млн км")
        
    print("\n" + "=" * 80)
    print("ТЕСТ ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 80)
    
else:
    print("\n✗ Нет результатов для анализа!")

# ============================================================================
# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

if results:
    print("\n8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("-" * 40)
    
    # Подготовка данных для сохранения
    save_data = {
        't3_utc_jd': [r['t3_utc_jd'] for r in results],
        't2_utc_jd': [r['t2_utc_jd'] for r in results],
        't1_utc_jd': [r['t1_utc_jd'] for r in results],
        'total_delay_sec': [r['total_delay_sec'] for r in results],
        'tau_up_sec': [r['tau_up_sec'] for r in results],
        'tau_down_sec': [r['tau_down_sec'] for r in results],
        'shapiro_sec': [r['shapiro_sec'] for r in results],
        'distance_up_km': [r['distance_up_km'] for r in results],
        'distance_down_km': [r['distance_down_km'] for r in results],
    }
    
    np.savez('light_time_results.npz', **save_data)
    print(f"✓ Результаты сохранены в light_time_results.npz")
    print(f"  Сохранено {len(results)} точек")