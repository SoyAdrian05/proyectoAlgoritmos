import numpy as np

#Constante que indica el numero de decimales de lso numero flotantes, se aplica la funcion round
NUMDECIMALES = 5

#clase para representar las tareas a organizar con los algoritmos
class Tarea(object):
    def __init__(self, ID, duration):
        self.id = ID
        self.duration = duration

    #metodo para usar la clase en el prinf
    def __str__(self):
        return f"Tarea (id: {self.id}, duration: {self.duration})"


# Clase para para construir las instancia para el problema
# contiene el atributo del numero de tareas y crea instancias de ese
# tamanio de acuerdo a al distribucion deseada
class generadorInstancias(object):

    def __init__(self):
        self.numTareas = 0
    
    def setNumTareas(self, numTareas):
        if numTareas <= 0 :
            raise ValueError("El número de tareas deben ser mayor a cero.")
        self.numTareas = numTareas

    #Metodo privado para convertir los datos generados por numpy
    # a un diccionario y que lo datos sean flotates en python
    # y les da el formato (numtarea, duracion)
    def __convierteDatos(self, datos):
        tareas = []
        i = 0
        for dato in (datos):
            # Use val.item() to convert most NumPy values to a native Python type:
            #tareas.append({"id":i,"duration":dato.item()})
            #tareas.append({"id":i,"duration":float(dato)})
            tareas.append(Tarea(i, round(float(dato), NUMDECIMALES)))
            i = i + 1
        return tareas

    def generaDatosExponencial(self, media):
        # Metodo para generar una instancia usando la distribucion exponencial

        # Se valida nque los parámetros sean correctos para evitar errores lógicos en ejecución.
        if media is None:
            raise ValueError("Exponencial -> Error en el parametro: 'media', null'")
        if media <= 0:
            raise ValueError("Exponencial -> Error en el parametro: 'media', Debe ser positivos.")

         #Se utiliza scale como parámetro de escala, el cual equivale a la media esperada.
        datosDistribucion = np.random.exponential(scale=media, size=self.numTareas)
        return self.__convierteDatos(datosDistribucion)

    def generaDatosUniforme(self, minParam, maxParam):
        if minParam is None:
            raise ValueError("Uniforme -> Error en el parametro: 'min'")
        if maxParam is None:
            raise ValueError("Uniforme -> Error en el parametro: 'max'")
        if minParam >= maxParam:
            raise ValueError("La duración mínima debe ser menor que la máxima.")
        
        datosDistribucion = np.random.uniform(low=minParam, high=maxParam, size=self.numTareas)
        return self.__convierteDatos(datosDistribucion)

    def generaDatosPareto(self, shape_alpha, min_duration, max_duration):
        if shape_alpha is None:
            raise ValueError("Pareto -> Error en el parametro: 'alpha'")
        if min_duration is None:
            raise ValueError("Pareto -> Error en el parametro: 'min'")
        if max_duration is None:
            raise ValueError("Pareto -> Error en el parametro: 'max'")

        # Se renombran variables para mayor claridad
        L = min_duration
        H = max_duration
        alpha = shape_alpha

        # Se precalculan términos para evitar recalcular potencias repetitivamente.
        # Como alpha > 0, se cumple que L^(-alpha) será mayor que H^(-alpha).
        high_bound_term = L ** (-alpha)
        low_bound_term = H ** (-alpha)

        # Se generan variables uniformes base, que sirven como "semilla" estadística.
        # U ∈ [0, 1]
        U = np.random.uniform(0, 1, self.numTareas)

        # Interpolación en el dominio transformado.
        # Esto mapea la uniformidad hacia el rango permitido de la Pareto.
        y = low_bound_term + U * (high_bound_term - low_bound_term)

        # Transformación inversa para obtener la muestra final.
        # Aquí se regresa del dominio de potencias al dominio original de duraciones.
        samples = y ** (-1 / alpha)

        # Ajuste por precisión numérica.
        # En ocasiones por floating point el valor puede salir levemente fuera.
        # El clip asegura que el resultado permanezca estrictamente acotado.
        datosDistribucion = np.clip(samples, min_duration, max_duration)

        return self.__convierteDatos(datosDistribucion)

# ==========================================
#   Metodo main (Pruebas)
# ==========================================

if __name__ == "__main__":
    # Configuración mínima para validar que todo compila y genera salida.
    N_TASKS = 10

    generador =  generadorInstancias()
    generador.setNumTareas(N_TASKS)

    print("--- Generando Instancia Exponencial ---")
    # Generamos carga con Exponencial
    workload = generador.generaDatosExponencial(15.0)

    # Mostramos las tareas
    print("Tareas (Exponencial):")
    for task in workload:
        print(task)

    print("--- Generando Instancia Pareto (Heavy Tail) ---")
    # Pareto acotada: simula cargas con outliers pero sin romper el rango.
    workload = generador.generaDatosPareto(1.2, 1.0, 500.0)

    print("Tareas (Pareto):")
    for task in workload:
        print(task)

    # Uniforma
    workload =  generador.generaDatosUniforme(2.0, 5.0)

    print("Tareas (Uniforme):")
    for task in workload:
        print(task)
