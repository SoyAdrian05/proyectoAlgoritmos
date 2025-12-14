import json
import numpy as np  # Se mantiene por consistencia con el resto del proyecto (aunque aquí no se use directamente).
import distributions as td  # Importamos el módulo previo para centralizar la lógica de muestreo.


def generate_workload(num_tasks, dist_type, **kwargs):
    # Función principal para construir una instancia de carga de trabajo.
    # La idea es: generar duraciones -> empaquetar en objetos -> devolver lista final.
    raw_durations = []

    # 1) Generar las duraciones crudas usando el módulo de distribuciones.
    # Se decide la distribución según dist_type (string).
    if dist_type == 'exponential':
        # Para exponencial se requiere 'mean' (media esperada / scale).
        mean = kwargs.get('mean')
        if mean is None:
            raise ValueError("Para exponencial se requiere el parámetro 'mean'")

        # Delegamos el muestreo al módulo td para no duplicar fórmulas aquí.
        raw_durations = td.generate_exponential_samples(num_tasks, mean)

    elif dist_type == 'pareto':
        # Para pareto acotada se requieren alpha, min y max.
        # alpha controla la cola pesada, min/max acotan el rango para estabilidad.
        alpha = kwargs.get('alpha')
        min_d = kwargs.get('min')
        max_d = kwargs.get('max')

        # Validación de presencia de parámetros.
        # Nota: se usa all(...) como chequeo rápido de completitud.
        if not all([alpha, min_d, max_d]):
            raise ValueError("Para pareto se requieren: 'alpha', 'min', 'max'")

        raw_durations = td.generate_bounded_pareto_samples(num_tasks, alpha, min_d, max_d)

    elif dist_type == 'uniform':
        # Uniforme: solo se requiere min y max.
        min_d = kwargs.get('min')
        max_d = kwargs.get('max')

        # Se valida que existan los parámetros esenciales.
        if min_d is None or max_d is None:
            raise ValueError("Para uniforme se requieren: 'min', 'max'")

        raw_durations = td.generate_uniform_samples(num_tasks, min_d, max_d)

    else:
        # Si llega aquí, significa que nos pidieron una distribución que no implementamos.
        # Mejor fallar temprano y claro.
        raise ValueError(f"Distribución '{dist_type}' no soportada.")

    # 2) Empaquetar en estructura final (lista de diccionarios).
    # Se asigna un ID incremental para trazabilidad y debugging.
    workload = []
    for i, duration in enumerate(raw_durations):
        task = {
            'id': i,

            # Convertimos a float nativo para evitar problemas de serialización.
            # (Numpy a veces trae tipos propios que JSON no entiende directo).
            'duration': float(duration),

            # Campos futuros (si luego quieres simular llegada, prioridad, etc.)
            # 'arrival_time': 0,
            # 'priority': 1
        }
        workload.append(task)

    # Retornamos la instancia completa lista para correr simulación o guardado.
    return workload


def save_instance_to_json(workload, filename):
    # Persistencia en JSON para reproducibilidad experimental.
    # Se usa indent para que sea legible por humanos también.
    with open(filename, 'w') as f:
        json.dump(workload, f, indent=4)
    print(f"Instancia guardada en: {filename}")


def load_instance_from_json(filename):
    # Carga de instancia previamente generada.
    # Útil para repetir experimentos sin regenerar aleatoriedad.
    with open(filename, 'r') as f:
        return json.load(f)


# ==========================================
# BLOQUE MAIN (Prueba de Generación)
# ==========================================
if __name__ == "__main__":
    # Configuración mínima para validar que todo compila y genera salida.
    N_TASKS = 10

    print("--- Generando Instancia Exponencial ---")
    # Generamos carga con Exponencial: típicamente modela tiempos tipo “servicio/espera”.
    workload_exp = generate_workload(N_TASKS, 'exponential', mean=15.0)

    # Mostramos algunas tareas para inspección visual rápida.
    print("Primeras 3 tareas (Exponencial):")
    for t in workload_exp[:3]:
        print(t)

    print("\n--- Generando Instancia Pareto (Heavy Tail) ---")
    # Pareto acotada: simula cargas con outliers pero sin romper el rango.
    workload_pareto = generate_workload(
        N_TASKS,
        'pareto',
        alpha=1.2,
        min=1.0,
        max=500.0
    )

    # Igual se imprimen las primeras tareas para ver que haya duraciones variadas.
    print("Primeras 3 tareas (Pareto):")
    for t in workload_pareto[:3]:
        print(t)

    # Guardado opcional: útil para reutilizar la misma instancia en varias corridas.
    # save_instance_to_json(workload_pareto, "test_pareto_instance.json")
