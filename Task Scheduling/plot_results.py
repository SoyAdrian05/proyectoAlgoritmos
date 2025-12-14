import matplotlib.pyplot as plt
import instance_generator as ig
import scheduler as sch  # Asegúrate de que scheduler tenga las funciones greedy y gurobipy
import numpy as np


def run_experiment_and_plot():
    # --- CONFIGURACIÓN DEL ESCENARIO ---
    # Usamos un escenario pequeño para que Gurobi pueda resolverlo rápido
    N_PROCESSORS = 4
    N_TASKS = 25

    print(f"--- Ejecutando Comparativa Visual ({N_TASKS} tareas, {N_PROCESSORS} CPUs) ---")

    # 1. Generamos una instancia "difícil" (Pareto / Cola Pesada)
    #    Una alta varianza suele mostrar mejor las diferencias entre algoritmos.
    tasks = ig.generate_workload(N_TASKS, 'pareto', alpha=1.1, min=1.0, max=50.0)

    # 2. Ejecutar los 3 Algoritmos
    print("Ejecutando Greedy List...")
    _, mk_list = sch.run_list_scheduling(tasks, N_PROCESSORS)

    print("Ejecutando LPT...")
    _, mk_lpt = sch.run_lpt_scheduling(tasks, N_PROCESSORS)

    print("Ejecutando Gurobi (Óptimo)...")
    try:
        # Time limit corto para la demo
        _, mk_opt = sch.run_optimal_gurobi(tasks, N_PROCESSORS, time_limit=10)
    except Exception as e:
        print(f"Gurobi no disponible o error: {e}")
        mk_opt = 0  # Valor dummy si falla

    # 3. Preparar datos para la gráfica
    algorithms = ['List Scheduling\n(Orden Llegada)', 'LPT\n(Ordenado)', 'Gurobi\n(Óptimo)']
    makespans = [mk_list, mk_lpt, mk_opt]

    # Colores: Rojo (peor), Azul (mejor), Verde (meta)
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    # 4. Crear la Gráfica
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, makespans, color=colors, edgecolor='black')

    # Añadir líneas de referencia
    if mk_opt > 0:
        plt.axhline(y=mk_opt, color='green', linestyle='--', alpha=0.5, label='Límite Óptimo')

    # Etiquetas y Títulos
    plt.ylabel('Makespan (Tiempo Total)', fontsize=12)
    plt.title(
        f'Comparación de Algoritmos de Asignación\n({N_TASKS} Tareas Pareto, {N_PROCESSORS} Procesadores)',
        fontsize=14)

    # Poner el valor exacto encima de cada barra
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Calcular la mejora de LPT sobre List
    improvement = ((mk_list - mk_lpt) / mk_list) * 100

    # Añadir caja de texto con conclusiones
    text_str = f"Mejora de LPT vs List: {improvement:.1f}%"
    plt.text(0.02, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend()
    plt.tight_layout()

    # Mostrar
    print("Generando gráfica...")
    plt.show()


if __name__ == "__main__":
    run_experiment_and_plot()