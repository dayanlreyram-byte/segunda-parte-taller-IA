"""
TALLER ANALITICO: LA VOTACION ESPACIAL
Asignatura: Inteligencia Artificial II - K-Vecinos Mas Cercanos (KNN)

Formula (Distancia Euclidiana en 2D):
    d(A, B) = raiz( (x2 - x1)^2 + (y2 - y1)^2 )

Dataset (X = Edad, Y = Salario en miles):
    Cliente 1: A(20, 30) -> NO COMPRA
    Cliente 2: B(40, 50) -> COMPRA
    Cliente 3: C(35, 45) -> COMPRA
Punto nuevo: (30, 40)
"""
from math import sqrt
from collections import Counter

# ---------------------------------------------------------------- datos
dataset = [
    ("A", (20, 30), "NO COMPRA"),
    ("B", (40, 50), "COMPRA"),
    ("C", (35, 45), "COMPRA"),
]
punto_nuevo = (30, 40)


# ------------------------------------------------------------ funciones
def distancia_euclidiana(p, q):
    """d(p, q) = raiz((x2 - x1)^2 + (y2 - y1)^2)"""
    return sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)


def knn(punto, datos, k):
    """Devuelve (clasificacion, vecinos, votos) para el K dado."""
    ordenados = sorted(datos, key=lambda d: distancia_euclidiana(punto, d[1]))
    vecinos = ordenados[:k]
    votos = Counter(clase for _, _, clase in vecinos)
    return votos.most_common(1)[0][0], vecinos, votos


# ------------------------------------------------------------- punto 1
print("=" * 62)
print("1. DISTANCIA EUCLIDIANA DESDE EL PUNTO NUEVO", punto_nuevo)
print("=" * 62)

x1, y1 = punto_nuevo
for nombre, (x2, y2), clase in dataset:
    dx, dy = x2 - x1, y2 - y1
    suma = dx ** 2 + dy ** 2
    d = sqrt(suma)
    print(f"\nd(Nuevo, {nombre}) = raiz[ ({x2}-{x1})^2 + ({y2}-{y1})^2 ]")
    print(f"              = raiz[ ({dx})^2 + ({dy})^2 ]")
    print(f"              = raiz[ {dx ** 2} + {dy ** 2} ]")
    print(f"              = raiz[ {suma} ]  ~= {d:.4f}   (clase: {clase})")

# ------------------------------------------------------------- punto 2
print("\n" + "=" * 62)
print("2. CLASIFICACION CON K = 1")
print("=" * 62)
clase1, vecinos1, votos1 = knn(punto_nuevo, dataset, 1)
for nombre, coord, clase in vecinos1:
    print(f"Vecino mas cercano: {nombre}{coord} -> {clase} "
          f"(d = {distancia_euclidiana(punto_nuevo, coord):.4f})")
print(f">> Clasificacion K=1: {clase1}")

# ------------------------------------------------------------- punto 3
print("\n" + "=" * 62)
print("3. CLASIFICACION CON K = 3")
print("=" * 62)
clase3, vecinos3, votos3 = knn(punto_nuevo, dataset, 3)
for nombre, coord, clase in vecinos3:
    print(f"Vecino {nombre}{coord} -> voto: {clase} "
          f"(d = {distancia_euclidiana(punto_nuevo, coord):.4f})")
print(f"Conteo de votos: {dict(votos3)}")
print(f">> Clasificacion K=3: {clase3}")

print("\n" + "-" * 62)
if clase1 == clase3:
    print(f"Hubo cambio de decision? NO. Con K=1 y K=3 el resultado es {clase1}.")
else:
    print(f"Hubo cambio de decision? SI: K=1 -> {clase1}, K=3 -> {clase3}.")
print("""
Analisis:
- C(35,45) es el vecino mas cercano (~7.07), y su clase es COMPRA.
- A y B estan empatados en distancia (~14.14), pero a lados opuestos
  del punto nuevo, y tienen clases distintas.
- Con K=3 se consulta a todo el dataset: 2 votos COMPRA vs 1 NO COMPRA,
  gana COMPRA por mayoria (K impar evita empates en la votacion).
""")
