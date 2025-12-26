import numpy as np
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt

# 1. КОНСТАНТЫ

GM_SUN     = 1.3271244004193938e11     # км^3 / c^2
GM_MERCURY = GM_SUN * 1.6601e-7
GM_EARTH   = GM_SUN * 3.0034896e-6

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


# 4. ФУНКЦИЯ ДЛЯ РИСОВАНИЯ КРУГОВЫХ ОРБИТ
def plot_circular_orbits_comparison(data):
    """Рисует сравнение круговых орбит MESSENGER относительно Меркурия"""
    
    print("\n" + "="*60)
    print("СРАВНЕНИЕ КРУГОВЫХ ОРБИТ")
    print("="*60)
    
    if isinstance(data, str):
        data = np.load(data)
    
    traj_sc = data['sc_positions']  # интегрированная траектория
    traj_eph = data['messenger_ephemeris']  # JPL
    traj_mer = data['mercury_positions']
    
    # переводим в систему отсчета Меркурия
    sc_rel = traj_sc - traj_mer
    eph_rel = traj_eph - traj_mer
    
    # 1. Находим плоскость орбиты методом PCA
    def fit_orbital_plane(points):
        """Находит плоскость орбиты через PCA"""
        # Центрируем точки
        center = np.mean(points, axis=0)
        centered = points - center
        
        # PCA для нахождения главных компонент
        U, s, Vt = np.linalg.svd(centered)
        
        # Нормаль к плоскости (наименьшая компонента)
        normal = Vt[2]
        
        # Два главных направления в плоскости орбиты
        u1 = Vt[0]  # Первый главный компонент
        u2 = Vt[1]  # Второй главный компонент
        
        # Среднее расстояние от центра
        distances = np.linalg.norm(centered, axis=1)
        mean_radius = np.mean(distances)
        
        # Эксцентриситет 
        eccentricity = 1 - s[1]/s[0] if s[0] > 0 else 0
        
        return center, normal, u1, u2, mean_radius, eccentricity
    
    # Анализ обеих траекторий
    sc_center, sc_normal, sc_u1, sc_u2, sc_mean_radius, sc_eccentricity = fit_orbital_plane(sc_rel)
    eph_center, eph_normal, eph_u1, eph_u2, eph_mean_radius, eph_eccentricity = fit_orbital_plane(eph_rel)
    
    print(f"\nИнтегрированная орбита:")
    print(f"  Средний радиус: {sc_mean_radius:.1f} км")
    print(f"  Эксцентриситет: {sc_eccentricity:.4f}")
    print(f"  Нормаль к плоскости: {sc_normal}")
    
    print(f"\nJPL:")
    print(f"  Средний радиус: {eph_mean_radius:.1f} км")
    print(f"  Эксцентриситет: {eph_eccentricity:.4f}")
    print(f"  Нормаль к плоскости: {eph_normal}")
    
    # 2. Строим круговые орбиты в реальных плоскостях
    def create_circular_orbit(center, u1, u2, radius, num_points=500):
        """Создает круговую орбиту в заданной плоскости"""
        theta = np.linspace(0, 2*np.pi, num_points)
        
        # параметрическое уравнение круга в плоскости орбиты
        # point = center + radius * (cosθ * u1 + sinθ * u2)
        orbit_points = np.zeros((num_points, 3))
        
        for i, angle in enumerate(theta):
            # Вектор в плоскости орбиты
            in_plane_vector = radius * (np.cos(angle) * u1 + np.sin(angle) * u2)
            orbit_points[i] = center + in_plane_vector
        
        return orbit_points
    
    # Создаем идеальные круговые орбиты
    sc_circle = create_circular_orbit(sc_center, sc_u1, sc_u2, sc_mean_radius)
    eph_circle = create_circular_orbit(eph_center, eph_u1, eph_u2, eph_mean_radius)
    
    # 3. Проецируем на лучшую плоскость для визуализации
    # Находим оси с наибольшей вариацией
    def find_best_plane(points):
        """Находит лучшую плоскость для проекции"""
        ranges = np.ptp(points, axis=0)
        sorted_indices = np.argsort(ranges)[::-1]
        return sorted_indices[:2]  # Две оси с наибольшим размахом
    
    best_axes = find_best_plane(sc_rel)
    axis_names = ['X', 'Y', 'Z']
    
    print(f"\nОси для отображения: {axis_names[best_axes[0]]} и {axis_names[best_axes[1]]}")
    
    # 4. Создаем графики
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1: Круговые орбиты в 2D проекции 
    ax1.plot(sc_circle[:, best_axes[0]], sc_circle[:, best_axes[1]], 
             'b-', linewidth=3, alpha=0.8, 
             label=f'Интегрированная (R={sc_mean_radius:.0f} км)')
    ax1.plot(eph_circle[:, best_axes[0]], eph_circle[:, best_axes[1]], 
             'r--', linewidth=3, alpha=0.8, 
             label=f'JPL (R={eph_mean_radius:.0f} км)')
    ax1.plot(0, 0, 'ko', markersize=10, label='Меркурий')
    ax1.plot(0, 0, 'yo', markersize=6)
    ax1.set_xlabel(f'{axis_names[best_axes[0]]} (км)')
    ax1.set_ylabel(f'{axis_names[best_axes[1]]} (км)')
    ax1.set_title('Идеальные круговые орбиты')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2: 3D вид орбит 
    ax2 = fig.add_subplot(122, projection='3d')
    
    # реальные траектории
    ax2.plot(sc_rel[:, 0], sc_rel[:, 1], sc_rel[:, 2], 
             'b-', linewidth=1, alpha=0.5, label='Интегрированная')
    ax2.plot(eph_rel[:, 0], eph_rel[:, 1], eph_rel[:, 2], 
             'r--', linewidth=1, alpha=0.5, label='JPL')
    
    # круговые орбиты
    ax2.plot(sc_circle[:, 0], sc_circle[:, 1], sc_circle[:, 2], 
             'b-', linewidth=2, alpha=0.8, label='Круг (интегр.)')
    ax2.plot(eph_circle[:, 0], eph_circle[:, 1], eph_circle[:, 2], 
             'r--', linewidth=2, alpha=0.8, label='Круг (JPL)')
    
    ax2.scatter([0], [0], [0], color='black', s=100, label='Меркурий')
    ax2.scatter([0], [0], [0], color='yellow', s=50)
    
    ax2.set_xlabel('X (км)')
    ax2.set_ylabel('Y (км)')
    ax2.set_zlabel('Z (км)')
    ax2.set_title('3D вид орбит и плоскостей')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('circular_orbits_comparison_corrected.png', dpi=150, bbox_inches='tight')
    print("\nСохранен график: circular_orbits_comparison_corrected.png")
    plt.show()
    
    
    
    return sc_circle, eph_circle


# 5. Запуск
def solve_direct_problem():
    print("="*60)
    print("Решение прямой задачи для MESSENGER")
    print("="*60)

    start_jd = 2457024.5
    end_jd   = 2457141.5
    dt_hours = 0.1     # шаг интегрирования 6 минут
    dt_days  = dt_hours / 24.0

    print(f"Период: {start_jd:.1f} – {end_jd:.1f} JD")
    print(f"Шаг: {dt_hours} часа")

    print("\n1. Загрузка векторных таблиц")
    base = "jpl_data"

    bodies = {}
    files = {
        "Sun":     (GM_SUN,     "sun.txt"),
        "Mercury": (GM_MERCURY, "mercury.txt"),
        "Earth":   (GM_EARTH,   "earth.txt"),
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
    r = messenger_pos(start_jd)
    v = messenger_vel(start_jd)
    
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

            traj_sc.append(r.copy())
            
            # позиции из Messenger для сравнения
            traj_messenger_eph.append(messenger_pos(jd).copy())

            mp = bodies["Mercury"][0](jd)
            ep = bodies["Earth"][0](jd)
            sp = bodies["Sun"][0](jd)

            traj_mer.append(mp.copy())
            traj_earth.append(ep.copy())

            dist_mercury.append(np.linalg.norm(r - mp))
            dist_earth.append(np.linalg.norm(r - ep))
            dist_sun.append(np.linalg.norm(r - sp))

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
    plot_circular_orbits_comparison(output_file)
    
    return True


if __name__ == "__main__":
    solve_direct_problem()