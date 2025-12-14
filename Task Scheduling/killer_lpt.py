import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scheduler as sch  # Tu módulo de algoritmos
import instance_generator as ig


def generate_graham_worst_case(m_processors):
    """
    Genera la secuencia exacta que causa el peor rendimiento teórico de LPT.
    Formula: 2m-1 tareas ordenadas + tareas pequeñas específicas.
    Teoría: R = 4/3 - 1/(3m)
    """
    tasks = []
    # Fase 1: Tareas decrecientes desde 2*m-1 hasta m
    # Ejemplo para m=5: 9, 8, 7, 6, 5
    for i in range(2 * m_processors - 1, m_processors - 1, -1):
        tasks.append(float(i))

    # Fase 2: Tareas de tamaño 'm' (tantas como sean necesarias para forzar el error)
    # Una tarea extra al final que rompe el equilibrio
    # En el ejemplo clásico se añaden tareas de tamaño específico
    # Simplificación efectiva: duplicamos las tareas "incómodas"

    # Para romper LPT visualmente, usaremos una variante más simple:
    # 2 procesadores. Tareas: 3, 3, 2, 2, 2
    # LPT: (3,2,2)=7, (3,2)=5. Makespan 7.
    # OPT: (3,3)=6, (2,2,2)=6. Makespan 6. Gap del 16%!

    if m_processors == 2:
        raw_durations = [3, 3, 2, 2, 2]
    else:
        # Generación "Semi-Killer" para M procesadores
        # Generamos tareas grandes muy parecidas pero no iguales
        # Sin tareas pequeñas.
        k = 3  # Multiplicador de tareas por procesador
        n_tasks = k * m_processors + 1
        # Uniforme entre 10 y 12 (muy bloqueadas)
        raw_durations = np.random.uniform(10.0, 15.0, n_tasks).tolist()

    # Empaquetar en formato diccionario
    workload = []
    for i, d in enumerate(raw_durations):
        workload.append({'id': i, 'duration': d})
    return workload


def run_stress_test():
    # --- ESCENARIO EXTREMO ---
    # Usamos POCOS procesadores para que sea más difícil equilibrar
    M_PROCESSORS = 3

    print(f"--- BUSCANDO ROMPER LPT (M={M_PROCESSORS}) ---")

    data = []

    # Vamos a probar 20 intentos de generar "Ladrillos Incómodos"
    # Usamos Uniforme[50, 60].
    # Al ser tareas tan grandes y similares, es difícil encajarlas perfecto.

    for i in range(20):
        # N = M * 2 + 1 suele ser el punto de dolor (ej. 7 tareas para 3 cpus)
        n_tasks = M_PROCESSORS * 2 + 1

        # Generamos manualmente una uniforme "fea" (sin tareas pequeñas)
        raw = np.random.randint(50, 75, n_tasks)  # Enteros para que sea más bloque
        tasks = [{'id': idx, 'duration': float(d)} for idx, d in enumerate(raw)]

        # Ejecutar LPT
        _, mk_lpt = sch.run_lpt_scheduling(tasks, M_PROCESSORS)

        # Ejecutar Gurobi
        try:
            _, mk_opt = sch.run_optimal_gurobi(tasks, M_PROCESSORS, time_limit=2)
        except:
            continue

        if mk_opt > 0 and mk_lpt > mk_opt:
            gap = (mk_lpt - mk_opt) / mk_opt * 100
            data.append({'Caso': i, 'LPT': mk_lpt, 'Optimo': mk_opt, 'Gap': gap})
            print(f"¡Caso {i} encontrado! Gap: {gap:.2f}% (LPT:{mk_lpt} vs Opt:{mk_opt})")
        else:
            print(f"Caso {i}: LPT encontró el óptimo.")

    if not data:
        print("LPT sigue ganando... ¡Es muy robusto!")
        return

    # Visualizar los casos donde LPT falló
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))

    # Gráfico de barras agrupado
    x = np.arange(len(df))
    width = 0.35

    plt.bar(x - width / 2, df['LPT'], width, label='LPT', color='blue', alpha=0.7)
    plt.bar(x + width / 2, df['Optimo'], width, label='Óptimo', color='green', alpha=0.7)

    plt.xlabel('ID del Experimento')
    plt.ylabel('Makespan')
    plt.title(f'Casos donde LPT falló (Uniforme "Ladrillos" [50-75], M={M_PROCESSORS})')
    plt.legend()

    # Anotar el Gap
    for i in range(len(df)):
        gap_val = df.iloc[i]['Gap']
        plt.text(i, df.iloc[i]['LPT'] + 1, f"+{gap_val:.1f}%", ha='center', color='red',
                 fontweight='bold')

    plt.ylim(bottom=min(df['Optimo']) * 0.8)  # Zoom en la parte superior
    plt.show()

    # --- DEMOSTRACIÓN MANUAL DEL CASO MATEMÁTICO ---
    print("\n--- EL CASO MATEMÁTICO DESTRUCTIVO ---")
    # Caso clásico: 2 procesadores, tareas [3, 3, 2, 2, 2]
    bad_tasks = [
        {'id': 1, 'duration': 3}, {'id': 2, 'duration': 3},
        {'id': 3, 'duration': 2}, {'id': 4, 'duration': 2}, {'id': 5, 'duration': 2}
    ]
    M_BAD = 2

    _, mk_bad_lpt = sch.run_lpt_scheduling(bad_tasks, M_BAD)
    _, mk_bad_opt = sch.run_optimal_gurobi(bad_tasks, M_BAD, time_limit=5)

    print(f"Tareas: [3, 3, 2, 2, 2] en 2 CPUs")
    print(f"LPT Makespan: {mk_bad_lpt}")
    print(f"Gurobi Makespan: {mk_bad_opt}")
    print(f"GAP REAL: {((mk_bad_lpt - mk_bad_opt) / mk_bad_opt) * 100:.2f}%")


if __name__ == "__main__":
    run_stress_test()