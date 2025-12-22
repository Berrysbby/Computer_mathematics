# light_time_use.py

from light_time_done import (
    solve_light_time_for_odf, 
    analyze_light_time_results,
    integrate_with_simulation,
    LightTimeSolver
)

# Просто импортируйте и используйте нужные функции
print("Импорт успешен! Доступны функции:")
print("1. solve_light_time_for_odf() - решение уравнений для ODF данных")
print("2. analyze_light_time_results() - анализ результатов")
print("3. integrate_with_simulation() - загрузка траекторий")
print("4. LightTimeSolver() - класс для решения уравнений")

# Пример использования:
if __name__ == "__main__":
    # Запуск полной обработки
    from light_time_done import main
    main()