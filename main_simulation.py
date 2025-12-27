import numpy as np
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt

# 1. КОНСТАНТЫ

GM_SUN     = 1.3271244004193938e11     # км^3 / c^2
GM_MERCURY = GM_SUN * 1.6601e-7
GM_EARTH   = GM_SUN * 3.0034896e-6
GM_VENUS   = GM_SUN * 2.4478383e-6
GM_JUPITER = GM_SUN * 9.547919e-4


R_MERCURY = 2439.7                     # км
AU_KM     = 1.495978707e8              # км


# 2. ЗАГРУЗКА JPL (SSB)

def load_jpl_vector_table(filename, body_name):
    print(f"Загрузка {filename}...")

    with open(filename) as f:
        content = f.read()

    start = content.find('$$SOE')
    end   = content.find('$$EOE')
    if start == -1 or end == -1:
        raise ValueError(f"Не найдены маркеры $$SOE / $$EOE в файле {filename}")

    lines = content[start+5:end].strip().splitlines()
    data = []

    for line in lines:
        parts = line.split(',')
        if len(parts) < 8:
            continue
        try:
            jd = float(parts[0])
            vals = [float(p) for p in parts[2:8]]
            data.append([jd, *vals])
        except:
            continue

    if len(data) == 0:
        raise ValueError(f"Нет данных в файле {filename}")

    data  = np.array(data)
    times = data[:, 0]
    pos   = data[:, 1:4]
    vel   = data[:, 4:7]

    print(f"  Загружено {len(times)} точек")
    print(f"  Диапазон JD: {times[0]:.2f} – {times[-1]:.2f}")

    pos_i = interp1d(times, pos, axis=0, bounds_error=False, fill_value="extrapolate")
    vel_i = interp1d(times, vel, axis=0, bounds_error=False, fill_value="extrapolate")

    return pos_i, vel_i, times[0], times[-1]



# 3. ДИНАМИКА
"""Вычисляет гравитационное ускорение КА от всех тел системы.    Ускорение КА в системе Меркурия
    r = r_sc - r_mercury
    Учитывает только гравитацию"""
def acceleration(r, jd, bodies):
    # положение Меркурия в SSB
    r_mer = bodies["Mercury"][0](jd)

    # положение КА в SSB
    r_sc = r_mer + r

    a = np.zeros(3)

    # центральное притяжение Меркурия
    d = np.linalg.norm(r)
    if d > 1e-6:
        a -= GM_MERCURY * r / d**3

    # возмущения от остальных тел
    for name, (pos_i, _, gm) in bodies.items():
        if name in ("Mercury", "Messenger"):
            continue

        r_i = pos_i(jd)

        dr_sc = r_i - r_sc
        dr_mer = r_i - r_mer

        d_sc = np.linalg.norm(dr_sc)
        d_mer = np.linalg.norm(dr_mer)

        if d_sc > 1e-6 and d_mer > 1e-6:
            a += gm * (dr_sc / d_sc**3 - dr_mer / d_mer**3)

    return a



def rk4_step(r, v, jd, dt_days, bodies):
    dt = dt_days * 86400.0

    k1r = v
    k1v = acceleration(r, jd, bodies)

    k2r = v + 0.5 * dt * k1v
    k2v = acceleration(r + 0.5 * dt * k1r, jd + 0.5 * dt_days, bodies)

    k3r = v + 0.5 * dt * k2v
    k3v = acceleration(r + 0.5 * dt * k2r, jd + 0.5 * dt_days, bodies)

    k4r = v + dt * k3v
    k4v = acceleration(r + dt * k3r, jd + dt_days, bodies)

    r_new = r + dt * (k1r + 2*k2r + 2*k3r + k4r) / 6
    v_new = v + dt * (k1v + 2*k2v + 2*k3v + k4v) / 6

    return r_new, v_new


# 4. ФУНКЦИЯ ДЛЯ РИСОВАНИЯ КРУГОВЫХ ОРБИТ
def plot_orbits_actual_coordinates(data):
    """Строит 3D график орбит и график разницы между расчетными и JPL данными"""
    
    if isinstance(data, str):
        data = np.load(data)
    
    traj_sc = data['sc_positions']      # наши посчитанные координаты
    traj_jpl = data['messenger_ephemeris']  # JPL координаты
    times = data['times']                # временные метки
    
    # Вычисляем разницу между расчетными и JPL данными
    pos_diff = np.linalg.norm(traj_sc - traj_jpl, axis=1)
    
    # Создаем фигуру с двумя subplots
    fig = plt.figure(figsize=(15, 6))
    
    # 1. 3D график орбит
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Строим наши посчитанные координаты (синяя линия)
    ax1.plot(traj_sc[:, 0], traj_sc[:, 1], traj_sc[:, 2], 
            'b-', linewidth=1.5, alpha=0.7, label='Наши расчеты')
    
    # Строим JPL координаты (красная линия)
    ax1.plot(traj_jpl[:, 0], traj_jpl[:, 1], traj_jpl[:, 2], 
            'r--', linewidth=1.5, alpha=0.7, label='JPL данные')
    
    # Начальные точки
    ax1.scatter(traj_sc[0, 0], traj_sc[0, 1], traj_sc[0, 2], 
               color='blue', s=30, marker='o', label='Начало (расчет)')
    ax1.scatter(traj_jpl[0, 0], traj_jpl[0, 1], traj_jpl[0, 2], 
               color='red', s=30, marker='s', label='Начало (JPL)')
    
    # Конечные точки
    ax1.scatter(traj_sc[-1, 0], traj_sc[-1, 1], traj_sc[-1, 2], 
               color='blue', s=30, marker='^', label='Конец (расчет)')
    ax1.scatter(traj_jpl[-1, 0], traj_jpl[-1, 1], traj_jpl[-1, 2], 
               color='red', s=30, marker='v', label='Конец (JPL)')
    
    ax1.set_xlabel('X (км)')
    ax1.set_ylabel('Y (км)')
    ax1.set_zlabel('Z (км)')
    ax1.set_title('Орбиты MESSENGER: наши расчеты vs JPL данные (SSB координаты)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. График разницы между расчетными и JPL данными
    ax2 = fig.add_subplot(122)
    
    # Преобразуем JD во время в днях от начала
    days_from_start = times - times[0]
    
    ax2.plot(days_from_start, pos_diff, 'g-', linewidth=2, alpha=0.8)
    ax2.fill_between(days_from_start, 0, pos_diff, alpha=0.3, color='green')
    
    # Добавляем горизонтальную линию для средней разницы
    mean_diff = pos_diff.mean()
    ax2.axhline(y=mean_diff, color='r', linestyle='--', alpha=0.7, 
                label=f'Средняя: {mean_diff:.2f} км')
    
    # Находим и отмечаем максимальную разницу
    max_diff = pos_diff.max()
    max_idx = np.argmax(pos_diff)
    ax2.scatter(days_from_start[max_idx], max_diff, color='red', s=50, zorder=5)
    ax2.annotate(f'Макс: {max_diff:.2f} км', 
                xy=(days_from_start[max_idx], max_diff),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax2.set_xlabel('Время от начала (дни)')
    ax2.set_ylabel('Разница позиций (км)')
    ax2.set_title('Разница между расчетными и JPL данными MESSENGER')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    
    plt.tight_layout()
    plt.savefig('orbits_comparison.png', dpi=150, bbox_inches='tight')
    print("\nСохранен график: orbits_comparison.png")
    print(f"  Средняя разница позиций: {mean_diff:.2f} км")
    print(f"  Максимальная разница позиций: {max_diff:.2f} км")
    plt.show()
    
    return traj_sc, traj_jpl, pos_diff

# 5. Запуск
def solve_direct_problem():
    print("Решение прямой задачи для MESSENGER")

    start_jd = 2457024.5
    end_jd   = 2457141.5
    dt_hours = 0.002     # шаг интегрирования 6 минут
    dt_days  = dt_hours / 24.0

    print(f"Период: {start_jd:.1f} – {end_jd:.1f} JD")
    print(f"Шаг: {dt_hours} часа")

    print("\n1. Загрузка векторных таблиц")
    base = "jpl_data"

    bodies = {}
    files = {
        "Sun":     (GM_SUN,     "sun.txt"),
        "Mercury": (GM_MERCURY, "mercury.txt"),
        "Venus":   (GM_VENUS,   "venus.txt"),
        "Earth":   (GM_EARTH,   "earth.txt"),
        "Jupiter": (GM_JUPITER, "jupiter.txt"),
        "Messenger": (0.0,      "messenger.txt")
    }

    for name, (gm, fname) in files.items():
        try:
            pos, vel, t0, t1 = load_jpl_vector_table(os.path.join(base, fname), name)
            bodies[name] = (pos, vel, gm)
        except Exception as e:
            print(f"  Ошибка загрузки {fname}: {e}")
            if name != "Messenger":
                raise e

    if "Messenger" not in bodies:
        raise FileNotFoundError("Файл messenger.txt не найден в директории jpl_data")
    
    print("\n2. Используем векторную таблицу MESSENGER из файла messenger.txt")
    
    #интерполяторы для Messenger
    messenger_pos = bodies["Messenger"][0]
    messenger_vel = bodies["Messenger"][1]
    
    # начальные условия
    r = messenger_pos(start_jd) - bodies["Mercury"][0](start_jd)
    v = messenger_vel(start_jd) - bodies["Mercury"][1](start_jd)

    
    # интегрирование
    print("\n3. Интегрирование")

    times        = []
    traj_sc     = []
    traj_mer    = []
    traj_earth  = []
    traj_messenger_eph = []  # траектория из Messenger

    dist_mercury = []
    dist_earth   = []
    dist_sun     = []

    jd = start_jd
    steps = int((end_jd - start_jd) / dt_days)

    for i in range(steps):
        if i % int(6/dt_hours) == 0:
            times.append(jd)

            r_mer = bodies["Mercury"][0](jd)
            traj_sc.append((r + r_mer).copy())

            
            # позиции из Messenger для сравнения
            traj_messenger_eph.append(messenger_pos(jd).copy())

            mp = bodies["Mercury"][0](jd)
            ep = bodies["Earth"][0](jd)
            sp = bodies["Sun"][0](jd)

            r_sc = r + mp  # КА в SSB

            dist_mercury.append(np.linalg.norm(r_sc - mp))
            dist_earth.append(np.linalg.norm(r_sc - ep))
            dist_sun.append(np.linalg.norm(r_sc - sp))

            traj_mer.append(mp.copy())
            traj_earth.append(ep.copy())

        r, v = rk4_step(r, v, jd, dt_days, bodies)
        jd += dt_days

    # анализ
    print("\n4. Анализ")

    dist_mercury = np.array(dist_mercury)  # Расстояние до Меркурия
    dist_earth   = np.array(dist_earth)    # Расстояние до Земли
    dist_sun     = np.array(dist_sun)      # Расстояние до Солнца

    # вычисляем разницу между интегрированной траекторией и данными из таблиц
    traj_sc_array = np.array(traj_sc)
    traj_messenger_eph_array = np.array(traj_messenger_eph)
    pos_diff = np.linalg.norm(traj_sc_array - traj_messenger_eph_array, axis=1)
    
    print(f"  Сравнение с эфемеридами Messenger:")
    print(f"    Средняя разница = {pos_diff.mean():.2f} км")
    print(f"    Максимальная разница = {pos_diff.max():.2f} км")
    
    print(f"\n  КА–Меркурий:")
    print(f"    min = {dist_mercury.min():.1f} км")
    print(f"    max = {dist_mercury.max():.1f} км")
    
    alts = dist_mercury - R_MERCURY
    print(f"  Высота над Меркурием:")
    print(f"    min = {alts.min():.1f} км")
    print(f"    max = {alts.max():.1f} км")

    print(f"\n  КА–Земля:")
    print(f"    min = {dist_earth.min()/1e6:.3f} млн км")
    print(f"    max = {dist_earth.max()/1e6:.3f} млн км")

    print(f"\n  КА–Солнце:")
    print(f"    min = {dist_sun.min()/1e6:.3f} млн км")
    print(f"    max = {dist_sun.max()/1e6:.3f} млн км")

    print("\n5. Сохранение")

    output_file = "direct_problem.npz"
    np.savez(
        output_file,
        times=np.array(times),
        sc_positions=np.array(traj_sc),
        messenger_ephemeris=np.array(traj_messenger_eph),
        mercury_positions=np.array(traj_mer),
        earth_positions=np.array(traj_earth),
        dist_sc_mercury=dist_mercury,
        dist_sc_earth=dist_earth,
        dist_sc_sun=dist_sun,
        position_differences=pos_diff,
        start_jd=start_jd,
        end_jd=end_jd,
        dt_hours=dt_hours
    )

    print(f"Обработка завершена. Файл с результатами: {output_file}")
    
    # Визуализация круговых орбит
    plot_orbits_actual_coordinates(output_file)
    
    return True


if __name__ == "__main__":
    solve_direct_problem()