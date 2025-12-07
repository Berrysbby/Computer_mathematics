import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from datetime import datetime
import re

# === Гравитационные параметры (м³/с²) ===
GM_sun      = 1.3271244e20
GM_mercury  = 2.2032e13
GM_earth    = 3.9860044e14

# === Путь к файлам ===
vector_files = {
    'mercury': 'mercury_vector.txt',
    'sun': 'sun_vector.txt',
    'earth': 'earth_vector.txt'
}

# === Парсинг векторной таблицы ===
def parse_vector_table(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    soe_index = content.find('$$SOE')
    eoe_index = content.find('$$EOE')
    
    data_str = content[soe_index + 5 : eoe_index].strip()
    lines = data_str.splitlines()
    
    t_grid = []
    pos_grid = []
    vel_grid = []
    
    epoch_1950 = datetime(1950, 1, 1)
    
    for line in lines:
        if not line or 'A.D.' not in line:
            continue
        
        parts = re.split(r',\s*', line.strip())
        jd = float(parts[0])
        
        # Секунды с 1950
        t_sec = (jd - 2433282.5) * 86400.0
        
        # Позиция и скорость
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        vx, vy, vz = float(parts[5]), float(parts[6]), float(parts[7])
        
        t_grid.append(t_sec)
        pos_grid.append([x, y, z])
        vel_grid.append([vx, vy, vz])
    
    t_grid = np.array(t_grid)
    pos_grid = np.array(pos_grid)
    vel_grid = np.array(vel_grid)
    
    print(f"Файл {filename}: {len(t_grid)} точек")
    return t_grid, pos_grid, vel_grid

# === Предзагрузка интерполяторов ===
def preload_planet_positions():
    interp_pos = {}
    
    for body, file in vector_files.items():
        t_grid, pos_grid, _ = parse_vector_table(file)
        
        interp_pos[body] = interp1d(t_grid, pos_grid.T, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    print("Интерполяция завершена")
    return interp_pos

# === Ускорение ===
def acceleration(t, state, interp_pos):
    pos_sc = state[:3]  # км
    
    merc_pos = interp_pos['mercury'](t)
    r_merc = (pos_sc - merc_pos) * 1e3  # м
    a = -GM_mercury * r_merc / np.linalg.norm(r_merc)**3
    
    for body in ['sun', 'earth']:
        body_pos = interp_pos[body](t)
        r_body_sc = (pos_sc - body_pos) * 1e3  # м
        r_body_merc = (merc_pos - body_pos) * 1e3  # м
        gm = GM_sun if body == 'sun' else GM_earth
        a += gm * (r_body_sc / np.linalg.norm(r_body_sc)**3 -
                   r_body_merc / np.linalg.norm(r_body_merc)**3)
    
    return np.concatenate([state[3:], a / 1e3])  # км/с²

# === Основная ===
if __name__ == "__main__":
    interp_pos = preload_planet_positions()
    
    t_start = 0.0
    t_end = 2592000.0  # 30 дней = 2.592e6 сек
    
    # Начальные условия для ПОЛЯРНОЙ орбиты (наклонение ~90°)
    initial_state = np.array([
        0.0,   # x = 0 км
        200.0, # y = 200 км
        0.0,   # z = 0 км
        0.0,   # vx = 0 км/с
        0.0,   # vy = 0 км/с
        10.0   # vz = 10 км/с — движение по Z!
    ])
    
    sol = solve_ivp(
        fun=lambda t, y: acceleration(t, y, interp_pos),
        t_span=(t_start, t_end),
        y0=initial_state,
        method='DOP853',
        rtol=1e-12,
        atol=1e-12
    )
    
    print(f"Интеграция завершена: {len(sol.t)} точек")
    
    # === 3D график ===
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    pos = sol.y[:3, :]
    ax.plot(pos[0], pos[1], pos[2], 'b-', linewidth=1, label='Орбита MESSENGER')
    ax.scatter(0, 0, 0, color='orange', s=100, label='Меркурий')
    
    ax.set_xlabel('X (км)')
    ax.set_ylabel('Y (км)')
    ax.set_zlabel('Z (км)')
    ax.set_title('Прямая задача: орбита MESSENGER за 30 дней\n(Меркурий + Солнце + Земля)')
    
    # Масштаб
    max_range = np.abs(pos).max() * 1.1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    ax.legend()
    plt.show()
    
    pos_interp = interp1d(sol.t, sol.y[:3], kind='cubic', axis=1)
    print("Пример позиции на t_end:", pos_interp(t_end))