import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scheduler as sch  # Tu módulo con Gurobi
import instance_generator as ig


def generate_killer_instance(m_processors):
    """
    Genera una instancia matemáticamente diseñada para romper LPT.
    Basada en la estructura [3, 3, 2, 2, 2] generalizada.

    Para M procesadores, generamos:
    - M tareas de tamaño 3
    - (M * 1.5) redondeado hacia arriba tareas de tamaño 2
    Esto fuerza a LPT a llenar con 3s y luego quedarse desbalanceado con los 2s.
    """
    tasks = []
    task_id = 0

    # 1. Generar M tareas de tamaño 3 (Las "Rocas")
    for _ in range(m_processors):
        tasks.append({'id': task_id, 'duration': 3.0})
        task_id += 1

    # 2. Generar tareas de tamaño 2 (Los "Ladrillos")
    # Cantidad justa para que sobre uno al final en LPT
    num_twos = int(m_processors * 1.5) + (1 if m_processors % 2 != 0 else 0)
    for _ in range(num_twos):
        tasks.append({'id': task_id, 'duration': 2.0})
        task_id += 1

    return tasks


def run_stress_test():
    # --- CONFIGURACIÓN DE LA ZONA DE PELIGRO ---
    M_PROCESSORS = 400

    # El punto débil de LPT suele ser N = 2*M + 1
    # Para M=4, N=9 es letal.
    N_TASKS = M_PROCESSORS * 2 + 1

    print(f"--- INICIANDO BUSQUEDA DE FALLOS (M={M_PROCESSORS}, N={N_TASKS}) ---")
    print("Buscando configuraciones donde LPT > Óptimo...")

    results = []

    # Intentos aleatorios buscando "Ladrillos Feos"
    # Usamos randint para generar bloques rígidos (ej: 5, 6, 7) sin decimales
    for i in range(15):
        # Generar carga "Bloque" (Uniforme Entera)
        raw_durations = np.random.randint(low=10, high=16, size=N_TASKS)
        tasks = [{'id': idx, 'duration': float(d)} for idx, d in enumerate(raw_durations)]

        # 1. Ejecutar LPT
        _, mk_lpt = sch.run_lpt_scheduling(tasks, M_PROCESSORS)

        # 2. Ejecutar Gurobi (Óptimo)
        try:
            _, mk_opt = sch.run_optimal_gurobi(tasks, M_PROCESSORS, time_limit=2)
        except Exception as e:
            print(f"Error Gurobi: {e}")
            continue

        # Comparar
        if mk_opt > 0:
            gap = (mk_lpt - mk_opt) / mk_opt * 100

            if gap > 0.001:  # Si hay diferencia real
                print(
                    f"  [!] CASO {i}: GAP ENCONTRADO -> {gap:.2f}% (LPT: {mk_lpt} vs Opt: {mk_opt})")
                results.append({
                    'Tipo': 'Aleatorio (Bloques)',
                    'Makespan_LPT': mk_lpt,
                    'Makespan_Opt': mk_opt,
                    'Gap': gap
                })
            else:
                # LPT lo hizo perfecto
                pass

    # --- AGREGAR EL CASO MATEMÁTICO DETERMINISTA ---
    print("\n--- EJECUTANDO EL 'MATHEMATICAL KILLER' ---")
    killer_tasks = generate_killer_instance(M_PROCESSORS)
    _, mk_k_lpt = sch.run_lpt_scheduling(killer_tasks, M_PROCESSORS)
    _, mk_k_opt = sch.run_optimal_gurobi(killer_tasks, M_PROCESSORS, time_limit=2)

    gap_k = (mk_k_lpt - mk_k_opt) / mk_k_opt * 100
    print(f"  [!!!] KILLER CASE: GAP -> {gap_k:.2f}% (LPT: {mk_k_lpt} vs Opt: {mk_k_opt})")

    results.append({
        'Tipo': 'Diseño Matemático',
        'Makespan_LPT': mk_k_lpt,
        'Makespan_Opt': mk_k_opt,
        'Gap': gap_k
    })

    # --- GRAFICAR RESULTADOS ---
    if not results:
        print("LPT sobrevivió a todo... ¡Es demasiado bueno hoy!")
        return

    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.35

    # Barras
    plt.bar(x - width / 2, df['Makespan_LPT'], width, label='LPT (Greedy)', color='#ff9999',
            edgecolor='black')
    plt.bar(x + width / 2, df['Makespan_Opt'], width, label='Gurobi (Óptimo)', color='#99ff99',
            edgecolor='black')

    # Etiquetas
    plt.ylabel('Makespan (Tiempo)')
    plt.title(f'Evidencia de Fallos en LPT (M={M_PROCESSORS})')
    plt.xticks(x, df['Tipo'] + "\nGap: " + df['Gap'].map('{:.1f}%'.format))
    plt.legend()

    # Poner valores sobre barras
    for i in range(len(df)):
        plt.text(x[i] - width / 2, df.iloc[i]['Makespan_LPT'], f"{df.iloc[i]['Makespan_LPT']:.1f}",
                 ha='center', va='bottom')
        plt.text(x[i] + width / 2, df.iloc[i]['Makespan_Opt'], f"{df.iloc[i]['Makespan_Opt']:.1f}",
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_stress_test()