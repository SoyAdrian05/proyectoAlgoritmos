import numpy as np


def generate_exponential_samples(num_samples, mean_duration):
    # Se valida que los parámetros sean correctos para evitar errores lógicos en ejecución.
    # (Porque si no, la distribución no tendría sentido matemático en este contexto).
    if num_samples <= 0 or mean_duration <= 0:
        raise ValueError("El número de muestras y la media deben ser positivos.")

    # Generación de muestras mediante distribución Exponencial.
    # Se utiliza scale como parámetro de escala, el cual equivale a la media esperada.
    # Esto nos permite simular tiempos tipo “espera” en procesos.
    return np.random.exponential(scale=mean_duration, size=num_samples)


def generate_uniform_samples(num_samples, min_duration, max_duration):
    # Genera muestras con distribución Uniforme para representar cargas "homogéneas".
    # Se recomienda para escenarios donde no se desea variabilidad extrema.
    if num_samples <= 0:
        raise ValueError("El número de muestras debe ser positivo.")

    # Validación de límites: el mínimo no puede superar al máximo.
    # De lo contrario el intervalo quedaría inconsistente.
    if min_duration >= max_duration:
        raise ValueError("La duración mínima debe ser menor que la máxima.")

    # Generación de valores aleatorios en el rango definido.
    # Nota: en teoría incluye el mínimo y no necesariamente incluye el máximo (dependiendo precisión).
    return np.random.uniform(low=min_duration, high=max_duration, size=num_samples)


def generate_bounded_pareto_samples(num_samples, shape_alpha, min_duration, max_duration):
    # Implementación de Pareto acotada.
    # Se realiza el procedimiento de inversión para muestrear la distribución correctamente.
    # (Esto permite modelar colas pesadas pero con límites para que no se vaya al infinito).
    if num_samples <= 0:
        raise ValueError("El número de muestras debe ser positivo.")

    # alpha debe ser positivo para que la función de potencia sea válida.
    # En la práctica, alpha controla qué tan agresiva es la cola.
    if shape_alpha <= 0:
        raise ValueError("shape_alpha debe ser > 0")

    # Se verifica el rango [min, max] para que sea un rango real.
    if min_duration >= max_duration:
        raise ValueError("La duración mínima debe ser menor que la máxima.")

    # Se renombran variables para mayor claridad conceptual.
    L = min_duration
    H = max_duration
    alpha = shape_alpha

    # Se precalculan términos para evitar recalcular potencias repetitivamente.
    # Como alpha > 0, se cumple que L^(-alpha) será mayor que H^(-alpha).
    high_bound_term = L ** (-alpha)
    low_bound_term = H ** (-alpha)

    # Se generan variables uniformes base, que sirven como "semilla" estadística.
    # U ∈ [0, 1]
    U = np.random.uniform(0, 1, num_samples)

    # Interpolación en el dominio transformado.
    # Esto mapea la uniformidad hacia el rango permitido de la Pareto.
    y = low_bound_term + U * (high_bound_term - low_bound_term)

    # Transformación inversa para obtener la muestra final.
    # Aquí se regresa del dominio de potencias al dominio original de duraciones.
    samples = y ** (-1 / alpha)

    # Ajuste por precisión numérica.
    # En ocasiones por floating point el valor puede salir levemente fuera.
    # El clip asegura que el resultado permanezca estrictamente acotado.
    return np.clip(samples, min_duration, max_duration)


def validate_stats(name, samples):
    # Función de validación exploratoria.
    # No garantiza veracidad estadística total, pero sirve como sanity check inicial.
    print(f"--- Validación: {name} ---")
    print(f"  Muestras: {len(samples)}")

    # Se imprimen mínimos y máximos observados para corroborar rango esperado.
    print(f"  Min: {np.min(samples):.6f}")
    print(f"  Max: {np.max(samples):.6f}")

    # Se imprimen tendencia central (media y mediana) para observar sesgo y cola.
    print(f"  Media: {np.mean(samples):.4f}")
    print(f"  Mediana: {np.median(samples):.4f}")
    print("-" * 30 + "\n")


if __name__ == "__main__":
    # Parámetro general: tamaño de muestra suficientemente grande para estabilidad.
    N = 10000

    # 1) Exponencial: típico para tiempos de llegada / servicio.
    exp_samples = generate_exponential_samples(N, 10.0)
    validate_stats("Exponencial", exp_samples)

    # 2) Pareto acotada: típico para cargas con outliers pero sin romper límites.
    alpha = 1.2
    min_val = 1.0
    max_val = 1000.0

    pareto_samples = generate_bounded_pareto_samples(N, alpha, min_val, max_val)
    validate_stats("Pareto Acotada", pareto_samples)

    # Validación con asserts: si falla, significa que algo está fuera de rango.
    # (En teoría no debería fallar gracias a clip, pero se deja como confirmación adicional).
    assert np.min(pareto_samples) >= min_val, f"Error: Min {np.min(pareto_samples)} < {min_val}"
    assert np.max(pareto_samples) <= max_val, f"Error: Max {np.max(pareto_samples)} > {max_val}"

    # 3) Uniforme: útil para simular bloques similares con variación controlada.
    print("Generando Uniforme entre 10 y 20...")
    unif_samples = generate_uniform_samples(1000, 10.0, 20.0)
    validate_stats("Uniforme [10, 20]", unif_samples)

    # Confirmaciones de rango. Ojo: uniform usa [min, max) pero por seguridad se deja <=.
    assert np.min(unif_samples) >= 10.0
    assert np.max(unif_samples) <= 20.0

    print(">>> Validaciones Uniforme exitosas.")
