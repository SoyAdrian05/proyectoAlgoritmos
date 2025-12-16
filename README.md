# Homogeneous Multiprocessor Scheduling Benchmark

Este repositorio aborda el Homogeneous Multiprocessor Scheduling Problem (HMSP), un problema NP-difícil que consiste en asignar un conjunto de tareas a múltiples procesadores idénticos, minimizando el makespan (tiempo total de ejecución).

## El proyecto implementa:
Generación de instancias aleatorias del problema, controlando:
- Número de tareas.
- Distribución del tiempo entre llegadas.
- Distribución de la duración de las tareas.

Solución exacta mediante programación matemática mixta usando el solver Gurobi.

Soluciones aproximadas mediante heurísticas voraces (greedy):
- Greedy no ordenado (online).
- Greedy ordenado (priorizando tareas largas).

Análisis comparativo de desempeño en términos de:
- Calidad de la solución (gap respecto a referencia).
- Tiempo de ejecución.

## Parámetros
- Número de procesadores: 10
- Número de tareas (N): 50, 100, 200, 400
- Repeticiones por escenario: 10
- Distribuciones: Exponencial, Pareto heavy-tail, Uniforme.
- Tiempo límite Gurobi: 5 segundos


1. Create a folder `example` in your home directory with several subfolders.

        user@computer:~$ cd ~/
        user@computer:~$ mkdir example
        user@computer:~$ cd example
        user@computer:~/example$ mkdir training_data holdout_data model holdout_outputs

## Autoress
1. José-Adrián Chávez-Olvera
2. Moisés Fernando Sagols Pacheco
3. Miguel de Jesús Vega Acevedo

