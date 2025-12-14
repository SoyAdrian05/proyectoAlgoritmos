import heapq
import copy
import instance_generator as ig
import gurobipy as gp
from gurobipy import GRB


# ==========================================================
# Algoritmos Greedy
# ==========================================================

def greedy_no_ordenado(tasks, num_processors):
    # siempre asigna la siguiente tarea al CPU "menos cargado".
    # Esto no garantiza óptimo, pero es más rápido

    # Heap de procesadores: (carga_actual, id_procesador, lista_tareas_asignadas)
    # Se usa heap porque sacar el menos cargado es O(log m) en vez de buscar a mano.
    processor_heap = [(0.0, i, []) for i in range(num_processors)]
    heapq.heapify(processor_heap)

    # Recorremos las tareas EN EL ORDEN QUE VIENEN (por eso no ordenado)
    for task in tasks:
        # Tomamos el procesador con menor carga actual
        current_load, p_id, assigned_tasks = heapq.heappop(processor_heap)

        # Guardamos el id de la tarea para trazabilidad
        assigned_tasks.append(task['id'])

        # Actualizamos la carga sumando la duración
        new_load = current_load + task['duration']

        # Metemos el procesador de regreso al heap con su nueva carga
        heapq.heappush(processor_heap, (new_load, p_id, assigned_tasks))

    # Convertimos heap -> dict para respuesta más amigable
    results = {}
    makespan = 0.0

    # El makespan es la carga máxima de todos los procesadores
    for load, p_id, task_ids in processor_heap:
        results[p_id] = {'load': load, 'tasks': task_ids}
        if load > makespan:
            makespan = load

    return results, makespan


def greedy_ordenado(tasks, num_processors):
    # Greedy Ordenado
    # Primero ordenamos tareas de mayor a menor duración,
    # luego aplicamos el greedy normal.
    # Nos ayuda porque resuelve tareas más pesadas primeero

    sorted_tasks = copy.deepcopy(tasks)

    # Orden descendente por duración (tarea larga primero).
    sorted_tasks.sort(key=lambda x: x['duration'], reverse=True)

    return greedy_no_ordenado(sorted_tasks, num_processors)


# ==========================================================
# Algoritmo exacto con Gurobi (para comparar contra óptimo)
# ==========================================================

def run_optimal_gurobi(tasks, num_processors, time_limit=60.0):
    # El modelo minimiza el makespan.
    # La variable C_max representa el tiempo en el que termina el procesador
    # más cargado.
    # Las restricciones obligan a que la carga de cada procesador
    # sea menor o igual a C_max.
    # Por lo tanto, C_max termina siendo el máximo de todas las cargas.
    # Al minimizar C_max, el modelo busca balancear las tareas
    # para que ningún procesador se quede trabajando mucho más que los demás.

    # Crear modelo
    m = gp.Model("ParallelScheduling")

    # Configuración típica del solver (modo silencioso para que no spamee consola).
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPGap', 0.01)  # 1% gap permitido (porque la vida es corta)

    print(f"   [Gurobi] Construyendo modelo para {len(tasks)} tareas y {num_processors} CPUs...")

    # Variable continua: C_max (lo que queremos minimizar)
    C_max = m.addVar(vtype=GRB.CONTINUOUS, name="C_max")

    # Variables binarias x[i,j] = 1 si tarea i se asigna al procesador j
    # Se arma un diccionario para manejarlo tipo matriz.
    x = {}
    for t in tasks:
        for j in range(num_processors):
            x[t['id'], j] = m.addVar(vtype=GRB.BINARY, name=f"x_{t['id']}_{j}")

    # Objetivo: minimizar C_max
    m.setObjective(C_max, GRB.MINIMIZE)

    # Restricción: cada tarea se asigna exactamente a 1 procesador
    for t in tasks:
        m.addConstr(
            gp.quicksum(x[t['id'], j] for j in range(num_processors)) == 1,
            name=f"assign_{t['id']}"
        )

    # Restricción: carga de cada procesador <= C_max
    for j in range(num_processors):
        load_expr = gp.quicksum(t['duration'] * x[t['id'], j] for t in tasks)
        m.addConstr(load_expr <= C_max, name=f"load_cap_{j}")

    # Optimizar
    m.optimize()

    # Procesar resultados
    results = {}
    final_makespan = 0.0

    # Aceptamos OPTIMAL o TIME_LIMIT (siempre que haya solución factible)
    if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
        if m.SolCount > 0:
            final_makespan = C_max.X
            print(f"   [Gurobi] Solución encontrada. Gap: {m.MIPGap:.2%}")

            # Inicializar estructura vacía
            for j in range(num_processors):
                results[j] = {'load': 0.0, 'tasks': []}

            # Reconstruir asignaciones leyendo x[i,j]
            for t in tasks:
                for j in range(num_processors):
                    # Umbral 0.5 por tolerancia numérica
                    if x[t['id'], j].X > 0.5:
                        results[j]['tasks'].append(t['id'])
                        results[j]['load'] += t['duration']
                        break
        else:
            print("   [Gurobi] Tiempo agotado sin solución factible.")
            return None, float('inf')
    else:
        print(f"   [Gurobi] No se encontró solución. Status: {m.Status}")
        return None, float('inf')

    return results, final_makespan


def print_stats(name, num_processors, makespan, assignments):
    # Impresión de métricas de comparacion.
    if assignments is None:
        print(f"--- {name}: FALLÓ O SIN SOLUCIÓN ---")
        return

    loads = [data['load'] for data in assignments.values()]
    avg_load = sum(loads) / len(loads)
    desbalance = max(loads) - min(loads)

    print(f"--- Resultados: {name} ---")
    print(f"  Makespan:    {makespan:.4f}")
    print(f"  Desbalance:  {desbalance:.4f}")
    print(f"  Eficiencia:  {avg_load / makespan:.2%}")
    print("-" * 40)


# ==========================================================
# MAIN (corrida de prueba)
# ==========================================================

if __name__ == "__main__":
    # Configuración del escenario (se mantiene moderado por Gurobi).
    M_PROCESSORS = 5
    N_TASKS = 50

    print(f"ESCENARIO: {N_TASKS} tareas, {M_PROCESSORS} procesadores.\n")

    # 1) Generar carga “difícil” con Pareto acotada (heavy tail controlada).
    tasks = ig.generate_workload(N_TASKS, 'pareto', alpha=1.2, min=1.0, max=100.0)

    # 2) Greedy sin ordenar
    res_greedy_no, mk_greedy_no = greedy_no_ordenado(tasks, M_PROCESSORS)
    print_stats("Greedy no ordenado", M_PROCESSORS, mk_greedy_no, res_greedy_no)

    # 3) Greedy ordenado (tipo LPT pero no le digamos así para que suene más “general”)
    res_greedy_ord, mk_greedy_ord = greedy_ordenado(tasks, M_PROCESSORS)
    print_stats("Greedy ordenado", M_PROCESSORS, mk_greedy_ord, res_greedy_ord)

    # 4) Óptimo con Gurobi
    try:
        res_opt, mk_opt = run_optimal_gurobi(tasks, M_PROCESSORS, time_limit=30)
        print_stats("Óptimo (Gurobi)", M_PROCESSORS, mk_opt, res_opt)

        # Comparativa (qué tan lejos quedó cada greedy del óptimo)
        if mk_opt > 0:
            gap_no = (mk_greedy_no - mk_opt) / mk_opt * 100
            gap_ord = (mk_greedy_ord - mk_opt) / mk_opt * 100

            print(f"\nCONCLUSIÓN:")
            print(f"Greedy no ordenado está a un {gap_no:.2f}% del óptimo.")
            print(f"Greedy ordenado está a un {gap_ord:.2f}% del óptimo.")

    except gp.GurobiError as e:
        print(f"\n[Error Gurobi]: {e}")
        print("Asegúrate de tener licencia instalada o reduce el tamaño del problema.")
