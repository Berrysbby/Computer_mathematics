import numpy as np
from scipy.interpolate import interp1d
import os

# 1. КОНСТАНТЫ

GM_SUN     = 1.3271244004193938e11     # км^3 / c^2
GM_MERCURY = GM_SUN * 1.6601e-7
GM_EARTH   = GM_SUN * 3.0034896e-6

R_MERCURY = 2439.7                     # км
AU_KM     = 1.495978707e8              # км


# 2. ЗАГРУЗКА ЭФЕМЕРИД JPL (SSB)

def load_jpl_vector_table(filename, body_name):
    print(f"Загрузка {filename}...")

    with open(filename) as f:
        content = f.read()

    start = content.find('$$SOE')
    end   = content.find('$$EOE')
    if start == -1 or end == -1:
        raise ValueError("Не найдены маркеры $$SOE / $$EOE")

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
"""Вычисляет гравитационное ускорение КА от всех тел системы.
    Учитывает только гравитацию"""
def acceleration(r, jd, bodies):
    a = np.zeros(3) # Вектор ускорения
    for pos_i, _, gm in bodies.values():
        dr = pos_i(jd) - r  # Вектор от КА к телу
        d  = np.linalg.norm(dr) # Расстояние
        if d > 1e-6:     # Защита от деления на 0
            a += gm * dr / d**3  # Вклад тела в ускорение
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


# 4. ПРЯМАЯ ЗАДАЧА

def solve_direct_problem():
    print("="*60)
    print("ПРЯМАЯ ЗАДАЧА: MESSENGER")
    print("="*60)

    start_jd = 2457024.5
    end_jd   = 2457141.5
    dt_hours = 0.1     #Шаг интегрирования 6 минут
    dt_days  = dt_hours / 24.0

    print(f"Период: {start_jd:.1f} – {end_jd:.1f} JD")
    print(f"Шаг: {dt_hours} часа")

    # Эфемериды
    print("\n1. Загрузка эфемерид...")
    base = "jpl_data"

    bodies = {}
    files = {
        "Sun":     (GM_SUN,     "sun.txt"),
        "Mercury": (GM_MERCURY, "mercury.txt"),
        "Earth":   (GM_EARTH,   "earth.txt"),
    }

    for name, (gm, fname) in files.items():
        pos, vel, t0, t1 = load_jpl_vector_table(os.path.join(base, fname), name)
        bodies[name] = (pos, vel, gm)

    # Начальные условия КА
    print("\n2. Начальные условия MESSENGER...")

    merc_pos = bodies["Mercury"][0](start_jd)
    merc_vel = bodies["Mercury"][1](start_jd)

    h_p = 200.0   # Перицентр (км над поверхностью)
    h_a = 10000.0   # Апоцентр (км над поверхностью)
    a   = (R_MERCURY + h_p + R_MERCURY + h_a) / 2  #Расчёт большой полуоси
    v_p = np.sqrt(GM_MERCURY * (2/(R_MERCURY+h_p) - 1/a))  #Скорость в перицентре:

    v_dir = merc_vel / np.linalg.norm(merc_vel)  #Направление скорости Меркурия
    r_dir = np.array([1.0, 0.0, 0.0])
    r_dir -= np.dot(r_dir, v_dir) * v_dir
    r_dir /= np.linalg.norm(r_dir)

    rel_r = r_dir * (R_MERCURY + h_p)  # Положение относительно Меркурия
    rel_v = np.cross(v_dir, r_dir)    # Векторное произведение для перпендикуляра
    rel_v = rel_v / np.linalg.norm(rel_v) * v_p # Нормализация и масштабирование

    r = merc_pos + rel_r  # Положение КА в SSB
    v = merc_vel + rel_v  # Скорость КА в SSB

    # Интегрирование
    print("\n3. Интегрирование...")

    times        = []
    traj_sc     = []
    traj_mer    = []
    traj_earth  = []

    dist_mercury = []
    dist_earth   = []

    jd = start_jd
    steps = int((end_jd - start_jd) / dt_days)

    for i in range(steps):
        if i % int(6/dt_hours) == 0:
            times.append(jd)

            traj_sc.append(r.copy())

            mp = bodies["Mercury"][0](jd)
            ep = bodies["Earth"][0](jd)

            traj_mer.append(mp.copy())
            traj_earth.append(ep.copy())

            dist_mercury.append(np.linalg.norm(r - mp))
            dist_earth.append(np.linalg.norm(r - ep))

        r, v = rk4_step(r, v, jd, dt_days, bodies)
        jd += dt_days

#Сохраняемые данные: Временные метки (JD), Положения КА, Меркурия, Земли, Расстояния КА-Меркурий и КА-Земля

    # Анализ
    print("\n4. Анализ...")

    dist_mercury = np.array(dist_mercury)  #Расстояние до Меркурия
    dist_earth   = np.array(dist_earth)  # Расстояние до Земли

    alts = dist_mercury - R_MERCURY  #Высота над поверхностью

    print(f"  КА–Меркурий:")
    print(f"    min = {dist_mercury.min():.1f} км")
    print(f"    max = {dist_mercury.max():.1f} км")

    print(f"  КА–Земля:")
    print(f"    min = {dist_earth.min()/1e6:.3f} млн км")
    print(f"    max = {dist_earth.max()/1e6:.3f} млн км")

    print(f"  Высота над Меркурием:")
    print(f"    min = {alts.min():.1f} км")
    print(f"    max = {alts.max():.1f} км")

    # Сохранение
    print("\n5. Сохранение...")

    np.savez(
        "direct_problem_results.npz",
        times=np.array(times),
        sc_positions=np.array(traj_sc),
        mercury_positions=np.array(traj_mer),
        earth_positions=np.array(traj_earth),
        dist_sc_mercury=dist_mercury,
        dist_sc_earth=dist_earth,
        start_jd=start_jd,
        end_jd=end_jd,
        dt_hours=dt_hours
    )

    print("Готово. Файл: direct_problem_results.npz")



if __name__ == "__main__":
    solve_direct_problem()
