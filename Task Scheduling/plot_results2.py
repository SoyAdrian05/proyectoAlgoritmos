import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

import instance_generator as ig
import scheduler as sch


def run_scientific_benchmark():
    # ==========================================
    # 1) INPUT (selección de distribución)
    # ==========================================
    print("\n--- CONFIGURACIÓN DEL BENCHMARK ---")
    print("Seleccione distribución:")
    print("   [1] Exponencial")
    print("   [2] Pareto (heavy tail)")
    print("   [3] Uniforme")
    opcion = input(">> Ingrese opción (1, 2 o 3): ").strip()

    # Se define distribución y parámetros estándar para no estar inventando en cada corrida.
    if opcion == '1':
        dist_type = 'exponential'
        dist_name = "Exponencial"
        dist_params = {'mean': 15.0}

    elif opcion == '2':
        dist_type = 'pareto'
        dist_name = "Pareto (Heavy Tail)"
        dist_params = {'alpha': 1.2, 'min': 1.0, 'max': 100.0}

    elif opcion == '3':
        dist_type = 'uniform'
        dist_name = "Uniforme [10-20]"
        dist_params = {'min': 10.0, 'max': 20.0}

    else:
        # Si el usuario se equivoca, se prefiere fallar aquí y no simular algo sin definición.
        raise ValueError("Opción inválida. Use 1, 2 o 3.")

    # ==========================================
    # 2) CONFIGURACIÓN EXPERIMENTAL
    # ==========================================
    N_PROCESSORS = 10
    TASK_COUNTS = [50, 100, 200, 400]  # 4 experimentos como se pidió
    REPETITIONS = 10
    GUROBI_TIME_LIMIT = 5

    data = []

    print(f"\n--- INICIANDO SIMULACIÓN: {dist_name} ---")
    print(f"Procesadores: {N_PROCESSORS}")
    print(f"Escenarios N: {TASK_COUNTS}")
    print(f"Parámetros: {dist_params}")
    print(f"Gurobi time limit: {GUROBI_TIME_LIMIT} s")
    print("-" * 60)

    # ==========================================
    # 3) LOOP PRINCIPAL
    # ==========================================
    for n in TASK_COUNTS:
        for _ in tqdm(range(REPETITIONS), desc=f"Simulando N={n}"):

            # Se genera una instancia por repetición para tener variabilidad real.
            tasks = ig.generate_workload(n, dist_type, **dist_params)

            # Greedy no ordenado (online): asigna al procesador con menor carga actual.
            _, mk_greedy_no = sch.greedy_no_ordenado(tasks, N_PROCESSORS)

            # Greedy ordenado: mismo greedy, pero las tareas se priorizan por duración.
            _, mk_greedy_ord = sch.greedy_ordenado(tasks, N_PROCESSORS)

            # Gurobi como referencia (óptimo o mejor encontrado en el tiempo límite).
            try:
                _, mk_ref = sch.run_optimal_gurobi(tasks, N_PROCESSORS, time_limit=GUROBI_TIME_LIMIT)
                ref_status = "OK"
            except Exception:
                # En caso de falla del solver, usamos un fallback para no romper el benchmark.
                mk_ref = mk_greedy_ord
                ref_status = "FALLBACK_GREEDY_ORD"

            # Gap contra la referencia mk_ref.
            # Si mk_ref no es óptimo real (por time limit), esto se interpreta como “vs referencia”.
            if mk_ref > 0:
                gap_no = (mk_greedy_no - mk_ref) / mk_ref * 100
                gap_ord = (mk_greedy_ord - mk_ref) / mk_ref * 100
            else:
                gap_no = 0.0
                gap_ord = 0.0

            # Se guarda todo en formato largo para agrupar fácil con pandas.
            data.append({'N': n, 'Algorithm': 'Greedy_No_Ordenado', 'Makespan': mk_greedy_no, 'Gap': gap_no, 'RefStatus': ref_status})
            data.append({'N': n, 'Algorithm': 'Greedy_Ordenado',    'Makespan': mk_greedy_ord, 'Gap': gap_ord, 'RefStatus': ref_status})
            data.append({'N': n, 'Algorithm': 'Gurobi_Referencia',  'Makespan': mk_ref,        'Gap': 0.0,    'RefStatus': ref_status})

    return pd.DataFrame(data), dist_name, N_PROCESSORS


def compute_summary_stats(df):
    # Agregación por N y algoritmo para obtener media y dispersión.
    # Se calcula un intervalo de confianza básico 95% usando 1.96.
    stats_df = df.groupby(['N', 'Algorithm']).agg(
        mean_makespan=('Makespan', 'mean'),
        std_makespan=('Makespan', 'std'),
        mean_gap=('Gap', 'mean'),
        std_gap=('Gap', 'std'),
        count=('Gap', 'count')
    ).reset_index()

    stats_df['std_makespan'] = stats_df['std_makespan'].fillna(0)
    stats_df['std_gap'] = stats_df['std_gap'].fillna(0)

    stats_df['ci_makespan'] = 1.96 * (stats_df['std_makespan'] / np.sqrt(stats_df['count']))
    stats_df['ci_gap'] = 1.96 * (stats_df['std_gap'] / np.sqrt(stats_df['count']))

    return stats_df


def plot_results(stats_df, dist_name, n_processors):
    # ==========================================================
    # PLOT 1: Makespan con doble eje Y (dos escalas de tiempo)
    # ==========================================================
    fig, ax_left = plt.subplots(figsize=(10, 6))

    # Eje izquierdo: Gurobi (queda abajo y se ve su variación real)
    subset_ref = stats_df[stats_df['Algorithm'] == 'Gurobi_Referencia']
    ax_left.errorbar(
        subset_ref['N'], subset_ref['mean_makespan'],
        yerr=subset_ref['ci_makespan'],
        label="Gurobi_Referencia (eje izq)",
        linestyle='-', marker='^', linewidth=1.8, capsize=4
    )
    ax_left.set_ylabel("Makespan (Gurobi / referencia)")
    ax_left.set_xlabel("Número de tareas (N)")
    ax_left.grid(True, alpha=0.3)

    # Eje derecho: Greedy (para que no se aplasten si son mucho más altos)
    ax_right = ax_left.twinx()

    for algo, style in [
        ('Greedy_No_Ordenado', {'linestyle': '--', 'marker': 'o', 'linewidth': 1.6}),
        ('Greedy_Ordenado',    {'linestyle': '-',  'marker': 's', 'linewidth': 3.2, 'alpha': 0.55}),
    ]:
        subset = stats_df[stats_df['Algorithm'] == algo]
        ax_right.errorbar(
            subset['N'], subset['mean_makespan'],
            yerr=subset['ci_makespan'],
            label=f"{algo} (eje der)",
            capsize=4,
            **style
        )

    ax_right.set_ylabel("Makespan (Greedy)")

    # Leyenda combinada (porque son dos ejes)
    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left")

    ax_left.set_title(f"Makespan con doble escala (dos ejes) - {dist_name}")
    plt.suptitle(f"Benchmark: {dist_name} ({n_processors} CPUs)", fontsize=12)
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # PLOT 2: Makespan normal (todo en un solo eje)
    # ==========================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, style in [
        ('Greedy_No_Ordenado', {'linestyle': '--', 'marker': 'o', 'linewidth': 1.6}),
        ('Greedy_Ordenado',    {'linestyle': '-',  'marker': 's', 'linewidth': 3.2, 'alpha': 0.55}),
        ('Gurobi_Referencia',  {'linestyle': '-',  'marker': '^', 'linewidth': 1.6}),
    ]:
        subset = stats_df[stats_df['Algorithm'] == algo]
        ax.errorbar(
            subset['N'], subset['mean_makespan'],
            yerr=subset['ci_makespan'],
            label=algo,
            capsize=4,
            **style
        )

    ax.set_title(f"Makespan promedio (eje único) - {dist_name}")
    ax.set_xlabel("Número de tareas (N)")
    ax.set_ylabel("Makespan (tiempo total)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # PLOT 3: Gap vs referencia (Gurobi = 0, se ven los greedy)
    # ==========================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, style in [
        ('Greedy_No_Ordenado', {'linestyle': '--', 'marker': 'o', 'linewidth': 1.8}),
        ('Greedy_Ordenado',    {'linestyle': '-',  'marker': 's', 'linewidth': 3.2, 'alpha': 0.55}),
    ]:
        subset = stats_df[stats_df['Algorithm'] == algo]
        ax.errorbar(
            subset['N'], subset['mean_gap'],
            yerr=subset['ci_gap'],
            label=algo,
            capsize=5,
            **style
        )

    ax.set_title(f"% desviación respecto a la referencia - {dist_name}")
    ax.set_xlabel("Número de tareas (N)")
    ax.set_ylabel("Gap (%) (peor que referencia)")
    ax.set_ylim(bottom=-0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df_results, distribution_name, n_procs = run_scientific_benchmark()

    print("\n--- RESUMEN (PROMEDIOS) ---")
    print(df_results.groupby(['N', 'Algorithm'])[['Makespan', 'Gap']].mean())

    stats_df = compute_summary_stats(df_results)
    plot_results(stats_df, distribution_name, n_procs)
