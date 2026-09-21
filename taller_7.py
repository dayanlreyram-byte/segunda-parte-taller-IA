"""
TALLER DE LABORATORIO: CLASIFICADOR UNIVERSAL (KNN con scikit-learn)
Asignatura: Inteligencia Artificial II

Mision practica:
  1. Transcribir el codigo base.
  2. Ampliar X_entrenamiento a >= 10 filas y agregar una tercera columna.
  3. Actualizar Y_entrenamiento para que coincida.
  4. Experimentar con n_neighbors=1 y n_neighbors=5.
  5. Responder la pregunta sobre la Maldicion de la Dimensionalidad.
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# =====================================================================
# PARTE 1. CODIGO BASE (transcrito del taller)
# =====================================================================
print("=" * 64)
print("PARTE 1. CODIGO BASE (3 puntos, 2 caracteristicas, K = 3)")
print("=" * 64)

# 1. Dataset de Entrenamiento: [Caracteristica 1, Caracteristica 2]
X_entrenamiento = np.array([
    [20, 30],  # Punto A
    [40, 50],  # Punto B
    [35, 45],  # Punto C
])

# Etiquetas: 0 = NO COMPRA, 1 = COMPRA
Y_entrenamiento = np.array([0, 1, 1])

# 2. Instanciar el modelo con K = 3
modelo_knn = KNeighborsClassifier(n_neighbors=3)

# 3. "Entrenar" (Memorizar los datos)
modelo_knn.fit(X_entrenamiento, Y_entrenamiento)

# 4. Predecir un nuevo punto
nuevo_cliente = np.array([[30, 40]])
prediccion = modelo_knn.predict(nuevo_cliente)

print("Clase predicha:", prediccion[0])

# =====================================================================
# PARTES 2 y 3. DATASET AMPLIADO (12 filas, 3 dimensiones)
# Columnas: [Edad, Salario en miles, Numero de Hijos]
# Etiquetas: 0 = NO COMPRA, 1 = COMPRA
# =====================================================================
X_entrenamiento = np.array([
    [20, 30, 0],  # 1
    [40, 50, 2],  # 2
    [36, 46, 1],  # 3
    [22, 28, 0],  # 4
    [25, 35, 1],  # 5
    [28, 36, 0],  # 6
    [45, 60, 3],  # 7
    [50, 70, 3],  # 8
    [32, 42, 2],  # 9
    [26, 33, 1],  # 10
    [38, 52, 2],  # 11
    [24, 31, 0],  # 12
])

Y_entrenamiento = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0])

# Verificacion: cada fila de X debe tener su etiqueta en Y
assert X_entrenamiento.shape == (12, 3)
assert len(X_entrenamiento) == len(Y_entrenamiento)

nuevo_cliente = np.array([[30, 40, 1]])  # Edad=30, Salario=40k, Hijos=1
nombres = {0: "NO COMPRA", 1: "COMPRA"}

# =====================================================================
# PARTE 4. EXPERIMENTO CON K
# =====================================================================
print("\n" + "=" * 64)
print("PARTE 4. EXPERIMENTO: K = 1 vs K = 5  (punto nuevo:",
      nuevo_cliente[0].tolist(), ")")
print("=" * 64)

resultados = {}
for k in (1, 5):
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_entrenamiento, Y_entrenamiento)
    clase = modelo.predict(nuevo_cliente)[0]
    resultados[k] = clase

    distancias, indices = modelo.kneighbors(nuevo_cliente)
    print(f"\n--- n_neighbors = {k} ---")
    for d, i in zip(distancias[0], indices[0]):
        print(f"  Fila {i + 1:>2} {X_entrenamiento[i].tolist()} "
              f"-> {nombres[Y_entrenamiento[i]]:<9} (d = {d:.3f})")
    print(f"  Probabilidades [NO COMPRA, COMPRA]: "
          f"{modelo.predict_proba(nuevo_cliente)[0]}")
    print(f"  >> Clase predicha: {clase} ({nombres[clase]})")

print("\n" + "-" * 64)
if resultados[1] != resultados[5]:
    print(f"La decision CAMBIO: K=1 -> {nombres[resultados[1]]}, "
          f"K=5 -> {nombres[resultados[5]]}.")
else:
    print(f"La decision NO cambio: {nombres[resultados[1]]}.")
print("""
Interpretacion:
- K=1 solo mira al vecino mas cercano (fila 9). Es muy sensible al ruido:
  un solo punto raro decide toda la clasificacion (sobreajuste).
- K=5 consulta a 5 vecinos y decide por mayoria. Es mas estable, y aqui
  la mayoria de los vecinos cercanos son clientes que NO compran.
- Regla de oro: K impar para evitar empates en la votacion binaria.
""")

# =====================================================================
# PARTE 5. MALDICION DE LA DIMENSIONALIDAD
# =====================================================================
print("=" * 64)
print("PARTE 5. MALDICION DE LA DIMENSIONALIDAD")
print("=" * 64)
print("""
Pregunta: si en lugar de 3 columnas tuvieran 1.000 (como los pixeles de
una imagen), que pasaria matematicamente con la Distancia Euclidiana?

Respuesta:
  d(p, q) = raiz( (p1-q1)^2 + (p2-q2)^2 + ... + (pn-qn)^2 )

  1. La suma acumula 1.000 terminos positivos, asi que las distancias
     crecen: d^2 crece proporcional a n, o sea d ~ raiz(n).
  2. Lo importante: la distancia al vecino MAS CERCANO y al MAS LEJANO
     se vuelven casi iguales. Los terminos se promedian (ley de los
     grandes numeros) y todas las distancias se concentran alrededor del
     mismo valor:
         (d_max - d_min) / d_min  --> 0   cuando n --> infinito
  3. Entonces "cercano" y "lejano" pierden significado. KNN ya no puede
     distinguir vecinos reales de puntos cualquiera, y la votacion se
     vuelve casi aleatoria.
  4. Ademas, el volumen del espacio crece exponencialmente con n: con
     1.000 dimensiones los datos quedan dispersos y se necesitarian
     cantidades enormes de muestras para "llenar" el espacio.
  5. Cada dimension pesa igual: las columnas irrelevantes o ruidosas
     aportan tanto a la distancia como las utiles.

  Soluciones tipicas: reducir dimensiones (PCA), seleccionar
  caracteristicas, escalar los datos o usar otras metricas.

Demostracion numerica (100 puntos aleatorios uniformes en [0,1]^n):
""")

rng = np.random.default_rng(42)
print(f"{'n (dim)':>8} | {'d_min':>8} | {'d_max':>8} | {'(dmax-dmin)/dmin':>17}")
print("-" * 50)
for n in (2, 3, 10, 100, 1000):
    puntos = rng.random((100, n))
    consulta = rng.random(n)
    d = np.linalg.norm(puntos - consulta, axis=1)
    contraste = (d.max() - d.min()) / d.min()
    print(f"{n:>8} | {d.min():>8.3f} | {d.max():>8.3f} | {contraste:>17.3f}")
print("\nA mayor dimension, el contraste tiende a 0: todos los puntos "
      "quedan a una distancia casi igual.")
