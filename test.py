import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ============ 1. КОНСТАНТЫ ============
GM_MERCURY = 1.3271244004193938e11 * 1.6601e-7  # 2.203e4 км³/с²
R_MERCURY = 2439.7  # км

# ============ 2. ТЕСТ РК4 С РАЗНЫМИ ШАГАМИ ============
def test_rk4_with_different_steps():
    """Тестируем РК4 с разными шагами"""
    print("="*60)
    print("ТЕСТ ТОЧНОСТИ РК4 С РАЗНЫМИ ШАГАМИ")
    print("="*60)
    
    # Параметры орбиты
    orbit_altitude = 5000.0
    a = R_MERCURY + orbit_altitude  # 7439.7 км
    v_orb = np.sqrt(GM_MERCURY / a)  # 1.721 км/с
    period = 2 * np.pi * a / v_orb  # секундах
    period_hours = period / 3600
    
    print(f"Параметры орбиты:")
    print(f"  Большая полуось: {a:.1f} км")
    print(f"  Орбитальная скорость: {v_orb:.3f} км/с")
    print(f"  Период: {period_hours:.2f} часов")
    
    # Тестируем разные шаги
    step_options = [
        (0.1, "0.1 часа (0.66% периода)"),
        (0.05, "0.05 часа (0.33% периода)"), 
        (0.02, "0.02 часа (0.13% периода)"),
        (0.01, "0.01 часа (0.07% периода)"),
    ]
    
    results = []
    
    for dt_hours, description in step_options:
        print(f"\n--- Тест: {description} ---")
        
        dt_days = dt_hours / 24.0
        dt_seconds = dt_hours * 3600.0
        
        # Интегрируем на 10 периодов
        n_periods = 10
        total_time = n_periods * period
        n_steps = int(total_time / dt_seconds)
        
        # Начальные условия
        pos = np.array([a, 0.0, 0.0])
        vel = np.array([0.0, v_orb, 0.0])
        
        # Простая функция ускорения (только Меркурий)
        def acceleration(position):
            r = np.linalg.norm(position)
            if r > 1e-6:
                return -GM_MERCURY * position / (r**3)
            return np.zeros(3)
        
        # Массивы для сохранения
        distances = []
        energies = []
        
        # Начальная энергия (на единицу массы)
        E0 = 0.5 * np.dot(vel, vel) - GM_MERCURY / np.linalg.norm(pos)
        
        # РК4 для одного тела
        for step in range(n_steps):
            # k1
            k1v = acceleration(pos) * dt_seconds
            k1r = vel * dt_seconds
            
            # k2
            k2v = acceleration(pos + 0.5*k1r) * dt_seconds
            k2r = (vel + 0.5*k1v) * dt_seconds
            
            # k3
            k3v = acceleration(pos + 0.5*k2r) * dt_seconds
            k3r = (vel + 0.5*k2v) * dt_seconds
            
            # k4
            k4v = acceleration(pos + k3r) * dt_seconds
            k4r = (vel + k3v) * dt_seconds
            
            # Обновление
            pos = pos + (k1r + 2*k2r + 2*k3r + k4r) / 6.0
            vel = vel + (k1v + 2*k2v + 2*k3v + k4v) / 6.0
            
            # Сохраняем каждые 100 шагов
            if step % 100 == 0:
                dist = np.linalg.norm(pos)
                distances.append(dist)
                
                # Энергия
                E = 0.5 * np.dot(vel, vel) - GM_MERCURY / dist
                energies.append(E)
        
        distances = np.array(distances)
        energies = np.array(energies)
        
        # Анализ
        dist_error = 100 * (np.max(distances) - np.min(distances)) / a
        energy_error = 100 * np.abs((energies - E0) / E0).max()
        
        print(f"  Шагов: {n_steps}")
        print(f"  Диапазон расстояний: {np.min(distances):.1f} - {np.max(distances):.1f} км")
        print(f"  Колебания расстояния: {dist_error:.3f}%")
        print(f"  Ошибка энергии: {energy_error:.6f}%")
        
        if dist_error < 0.1:
            rating = "✅ ОТЛИЧНО"
        elif dist_error < 1.0:
            rating = "⚠ ХОРОШО" 
        elif dist_error < 5.0:
            rating = "❌ ПЛОХО"
        else:
            rating = "💀 КАТАСТРОФА"
        
        results.append((dt_hours, dist_error, energy_error, rating))
    
    # Вывод результатов
    print("\n" + "="*60)
    print("ИТОГИ ТЕСТИРОВАНИЯ:")
    print("="*60)
    print("Шаг (часы) | % периода | Колебания | Ошибка энергии | Оценка")
    print("-"*60)
    
    for dt_hours, dist_err, energy_err, rating in results:
        percent_of_period = 100 * dt_hours / period_hours
        print(f"{dt_hours:7.3f}   | {percent_of_period:7.2f}%  | {dist_err:8.3f}% | {energy_err:12.6f}% | {rating}")
    
    return results

# ============ 3. ИСПРАВЛЕННАЯ МОДЕЛЬ С МАЛЕНЬКИМ ШАГОМ ============
def accurate_mercury_only_simulation():
    """Точная модель с маленьким шагом"""
    print("\n" + "="*60)
    print("ТОЧНАЯ МОДЕЛЬ ТОЛЬКО С МЕРКУРИЕМ")
    print("="*60)
    
    GM = GM_MERCURY
    a = R_MERCURY + 5000.0  # 7439.7 км
    v0 = np.sqrt(GM / a)  # 1.721 км/с
    
    print(f"GM Меркурия: {GM:.3e} км³/с²")
    print(f"Большая полуось: {a:.1f} км")
    print(f"Начальная скорость: {v0:.3f} км/с")
    print(f"Период: {2*np.pi*a/v0/3600:.2f} часов")
    
    # ОЧЕНЬ маленький шаг! 1% от периода
    period = 2 * np.pi * a / v0  # секундах
    dt_seconds = period * 0.01  # 1% от периода
    dt_hours = dt_seconds / 3600
    
    print(f"\nШаг интегрирования:")
    print(f"  {dt_seconds:.1f} секунд = {dt_hours:.3f} часов")
    print(f"  {100*dt_hours/(period/3600):.1f}% от периода")
    
    # Интегрируем на 10 периодов
    n_periods = 10
    total_time = n_periods * period
    n_steps = int(total_time / dt_seconds)
    
    print(f"Интегрирование на {n_periods} периодов:")
    print(f"  Всего шагов: {n_steps}")
    
    # Начальные условия
    pos = np.array([a, 0.0, 0.0])
    vel = np.array([0.0, v0, 0.0])
    
    def acceleration(position):
        r = np.linalg.norm(position)
        return -GM * position / (r**3)
    
    # Массивы для сохранения
    positions = []
    velocities = []
    distances = []
    times = []
    
    current_time = 0.0
    
    for step in range(n_steps + 1):
        # Сохраняем состояние
        if step % 10 == 0:  # каждые 10 шагов
            positions.append(pos.copy())
            velocities.append(vel.copy())
            distances.append(np.linalg.norm(pos))
            times.append(current_time)
        
        # Шаг РК4
        if step < n_steps:
            # k1
            k1v = acceleration(pos) * dt_seconds
            k1r = vel * dt_seconds
            
            # k2
            k2v = acceleration(pos + 0.5*k1r) * dt_seconds
            k2r = (vel + 0.5*k1v) * dt_seconds
            
            # k3
            k3v = acceleration(pos + 0.5*k2r) * dt_seconds
            k3r = (vel + 0.5*k2v) * dt_seconds
            
            # k4
            k4v = acceleration(pos + k3r) * dt_seconds
            k4r = (vel + k3v) * dt_seconds
            
            # Обновление
            pos = pos + (k1r + 2*k2r + 2*k3r + k4r) / 6.0
            vel = vel + (k1v + 2*k2v + 2*k3v + k4v) / 6.0
        
        current_time += dt_seconds
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    distances = np.array(distances)
    times = np.array(times) / 3600  # в часы
    
    # Анализ
    print(f"\nРЕЗУЛЬТАТЫ:")
    print(f"Минимальное расстояние: {np.min(distances):.3f} км")
    print(f"Максимальное расстояние: {np.max(distances):.3f} км")
    print(f"Среднее расстояние: {np.mean(distances):.3f} км")
    print(f"Начальное расстояние: {a:.3f} км")
    
    dist_error = 100 * np.abs(distances - a).max() / a
    print(f"Максимальная ошибка расстояния: {dist_error:.6f}%")
    
    # Энергия
    energies = 0.5 * np.sum(velocities**2, axis=1) - GM / distances
    energy_error = 100 * np.abs((energies - energies[0]) / energies[0]).max()
    print(f"Максимальная ошибка энергии: {energy_error:.6f}%")
    
    if dist_error < 0.01:
        print("✅ ОТЛИЧНО: Орбита практически круговая!")
    elif dist_error < 0.1:
        print("⚠ ХОРОШО: Небольшие колебания")
    else:
        print("❌ ПЛОХО: Орбита нестабильна")
    
    # Визуализация
    plot_accurate_results(times, positions, distances, velocities, a, v0, GM)
    
    return positions, distances, velocities

def plot_accurate_results(times, positions, distances, velocities, a, v0, GM):
    """Визуализация точных результатов"""
    fig = plt.figure(figsize=(14, 10))
    
    # 1. XY траектория
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, linewidth=1)
    ax1.scatter(0, 0, color='red', s=100)
    
    # Идеальная окружность
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = a * np.cos(theta)
    circle_y = a * np.sin(theta)
    ax1.plot(circle_x, circle_y, 'r--', alpha=0.5, linewidth=0.5, label='Идеальная окружность')
    
    ax1.set_xlabel('X (км)')
    ax1.set_ylabel('Y (км)')
    ax1.set_title('Траектория КА')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. Расстояние от центра
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(times, distances, 'g-', linewidth=2)
    ax2.axhline(y=a, color='r', linestyle='--', alpha=0.5, label=f'Теоретическое: {a:.1f} км')
    ax2.set_xlabel('Время (часы)')
    ax2.set_ylabel('Расстояние (км)')
    ax2.set_title('Расстояние от центра Меркурия')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Ошибка расстояния
    ax3 = fig.add_subplot(2, 3, 3)
    error = 100 * (distances - a) / a
    ax3.plot(times, error, 'm-', linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Время (часы)')
    ax3.set_ylabel('Ошибка (%)')
    ax3.set_title('Относительная ошибка расстояния')
    ax3.grid(True, alpha=0.3)
    
    # 4. Скорость
    ax4 = fig.add_subplot(2, 3, 4)
    speeds = np.linalg.norm(velocities, axis=1)
    ax4.plot(times, speeds, 'r-', linewidth=2)
    ax4.axhline(y=v0, color='b', linestyle='--', alpha=0.5, label=f'Теоретическая: {v0:.3f} км/с')
    ax4.set_xlabel('Время (часы)')
    ax4.set_ylabel('Скорость (км/с)')
    ax4.set_title('Скорость КА')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Энергия
    ax5 = fig.add_subplot(2, 3, 5)
    energies = 0.5 * speeds**2 - GM / distances
    energy_error = 100 * (energies - energies[0]) / np.abs(energies[0])
    ax5.plot(times, energy_error, 'c-', linewidth=1)
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.set_xlabel('Время (часы)')
    ax5.set_ylabel('Ошибка энергии (%)')
    ax5.set_title('Сохранение энергии')
    ax5.grid(True, alpha=0.3)
    
    # 6. Фазовый портрет (r vs v)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(distances, speeds, c=times, cmap='viridis', s=10, alpha=0.7)
    ax6.axhline(y=v0, color='b', linestyle='--', alpha=0.5)
    ax6.axvline(x=a, color='r', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Расстояние (км)')
    ax6.set_ylabel('Скорость (км/с)')
    ax6.set_title('Фазовый портрет (r vs v)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============ 4. РЕКОМЕНДАЦИИ ДЛЯ ПОЛНОЙ МОДЕЛИ ============
def get_recommendations():
    """Рекомендации для полной модели с Солнцем и Меркурием"""
    print("\n" + "="*60)
    print("РЕКОМЕНДАЦИИ ДЛЯ ПОЛНОЙ МОДЕЛИ")
    print("="*60)
    
    print("\n1. ШАГ ИНТЕГРИРОВАНИЯ:")
    print("   Для орбиты вокруг Меркурия (период ~7.55 часов):")
    print("   - Минимум: 0.1 часа (6.6% периода)")
    print("   - Рекомендуется: 0.02 часа (1.3% периода)")
    print("   - Для высокой точности: 0.01 часа (0.7% периода)")
    
    print("\n2. ДЛИТЕЛЬНОСТЬ ИНТЕГРИРОВАНИЯ:")
    print("   - Реальный период MESSENGER: ~12 часов")
    print("   - Шаг 0.02 часа = 720 шагов на период")
    print("   - 117 дней = 2808 часов = 140400 шагов (при шаге 0.02 часа)")
    
    print("\n3. УЧЕТ СОЛНЦА:")
    print("   - GM Солнца: 1.327e11 км³/с²")
    print("   - GM Меркурия: 2.203e4 км³/с²")
    print("   - Отношение: 6,000,000:1")
    print("   - Но! На расстоянии 7439 км от Меркурия:")
    print("     * Ускорение от Меркурия: 3.98e-6 км/с²")
    print("     * Ускорение от Солнца: 3.68e-6 км/с²")
    print("     * Они сравнимые!")
    
    print("\n4. ВЫВОД:")
    print("   - Нужен ОЧЕНЬ маленький шаг (0.01-0.02 часа)")
    print("   - Интегрирование будет долгим (140K+ шагов)")
    print("   - КА будет медленно дрейфовать из-за Солнца")
    print("   - Это НОРМАЛЬНО для упрощенной модели")

# ============ 5. ГЛАВНАЯ ФУНКЦИЯ ============
def main():
    print("="*60)
    print("АНАЛИЗ ТОЧНОСТИ ИНТЕГРИРОВАНИЯ")
    print("="*60)
    
    print("\nПроблема: шаг интегрирования слишком большой для точной орбиты.")
    
    print("\nВыберите тест:")
    print("1. Тест точности РК4 с разными шагами")
    print("2. Точная модель с маленьким шагом")
    print("3. Рекомендации для полной модели")
    
    choice = input("Введите 1, 2 или 3: ").strip()
    
    if choice == "1":
        test_rk4_with_different_steps()
    elif choice == "2":
        accurate_mercury_only_simulation()
    elif choice == "3":
        get_recommendations()
    else:
        print("Неверный выбор. Запускаю тест точности...")
        test_rk4_with_different_steps()

# ============ 6. ЗАПУСК ============
if __name__ == "__main__":
    main()