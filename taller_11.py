"""
TALLER DE LABORATORIO: EXPLORANDO LAS MATRICES
Asignatura: Inteligencia Artificial II - Red neuronal (forward pass)

Mision practica:
  1. Copiar el codigo, ejecutarlo y observar la probabilidad.
  2. Imprimir Z1 y A1 y analizar como la sigmoide transformo los
     valores puros de Z1 al rango (0, 1).
  3. El reto dimensional: procesar 2 clientes al mismo tiempo (lote /
     batch) cambiando X a una matriz de 2x3.
  4. Ejecutar: np.dot calcula las salidas de los dos clientes
     simultaneamente sin cambiar las matrices de pesos.
"""
import numpy as np


# Funcion de Activacion: Sigmoide (devuelve un valor entre 0 y 1)
def sigmoide(x):
    return 1 / (1 + np.exp(-x))


# =====================================================================
# PARTE 1. CODIGO BASE (transcrito del taller)
# =====================================================================
print("=" * 64)
print("PARTE 1. CODIGO BASE: 1 CLIENTE, 3 CARACTERISTICAS")
print("=" * 64)

# 1. ENTRADA (X): 1 cliente con 3 caracteristicas
X = np.array([0.5, 0.8, 0.2])

# 2. CAPA OCULTA (4 Neuronas)
# Matriz W1 de (3 entradas x 4 neuronas)
W1 = np.array([
    [0.1, 0.2, -0.3, 0.4],
    [-0.5, 0.6, 0.7, -0.8],
    [0.9, -0.1, 0.2, 0.3]
])
b1 = np.array([0.1, -0.2, 0.3, -0.4])  # 4 Sesgos

# --- PROCESO CAPA OCULTA ---
Z1 = np.dot(X, W1) + b1
A1 = sigmoide(Z1)  # Salida de la capa oculta

# 3. CAPA DE SALIDA (1 Neurona)
# Matriz W2 de (4 entradas ocultas x 1 neurona final)
W2 = np.array([0.5, -0.6, 0.7, 0.8])
b2 = np.array([-0.1])

# --- PROCESO CAPA FINAL ---
Z2 = np.dot(A1, W2) + b2
Salida_Final = sigmoide(Z2)

print("Prediccion de la Red (Probabilidad):", np.round(Salida_Final[0], 4))

# =====================================================================
# PARTE 2. ANALISIS DE Z1 Y A1
# =====================================================================
print("\n" + "=" * 64)
print("PARTE 2. Z1 (valores puros) vs A1 (despues de la sigmoide)")
print("=" * 64)
print("X  =", X, " forma", X.shape)
print("W1 forma", W1.shape, "| b1 forma", b1.shape)
print("\nZ1 = X . W1 + b1 =", np.round(Z1, 4), " forma", Z1.shape)
print("A1 = sigmoide(Z1) =", np.round(A1, 4), " forma", A1.shape)

print("\nNeurona |     Z1     ->     A1    | lectura")
print("-" * 58)
for i, (z, a) in enumerate(zip(Z1, A1), start=1):
    if z > 0:
        lectura = "Z1 positivo -> A1 > 0.5 (neurona 'activa')"
    elif z < 0:
        lectura = "Z1 negativo -> A1 < 0.5 (neurona 'inactiva')"
    else:
        lectura = "Z1 = 0 -> A1 = 0.5"
    print(f"   {i}    | {z:>8.4f}   -> {a:>8.4f}  | {lectura}")

print("""
Analisis:
- Z1 son numeros "puros" sin limite: pueden ser negativos, positivos o
  enormes segun los pesos. Aqui van de -0.78 a 0.75, pero con otros
  datos podrian valer -50 o +50.
- La sigmoide 1 / (1 + e^-z) los "aplasta" al rango (0, 1):
    z muy negativo -> salida cerca de 0
    z = 0          -> salida exactamente 0.5
    z muy positivo -> salida cerca de 1
- Conserva el orden (es monotona creciente): el Z1 mas grande sigue
  siendo el A1 mas grande, pero comprimido. Cerca de 0 la curva es
  casi recta; en los extremos se satura (se aplana hacia 0 o 1).
- Sin activacion no lineal, las dos capas colapsarian en una sola
  multiplicacion de matrices: la red seria equivalente a un modelo
  lineal simple. La sigmoide es lo que le da poder de aprendizaje.
- La salida final tambien pasa por sigmoide, por eso se lee como una
  probabilidad entre 0 y 1.
""")

# =====================================================================
# PARTES 3 y 4. EL RETO DIMENSIONAL: PROCESAMIENTO EN LOTE (BATCH)
# =====================================================================
print("=" * 64)
print("PARTES 3 y 4. PROCESAMIENTO EN LOTE: 2 CLIENTES A LA VEZ")
print("=" * 64)

# Solo cambia X: ahora es una matriz de 2x3 (2 clientes x 3 caracteristicas)
X = np.array([[0.5, 0.8, 0.2], [0.1, 0.9, 0.9]])

# --- PROCESO CAPA OCULTA (mismo codigo, mismas matrices) ---
Z1 = np.dot(X, W1) + b1
A1 = sigmoide(Z1)

# --- PROCESO CAPA FINAL (mismo codigo, mismas matrices) ---
Z2 = np.dot(A1, W2) + b2
Salida_Final = sigmoide(Z2)

print("X  forma", X.shape, "(2 clientes x 3 caracteristicas)")
print("\nZ1 forma", Z1.shape, "(2 clientes x 4 neuronas ocultas)")
print(np.round(Z1, 4))
print("\nA1 forma", A1.shape)
print(np.round(A1, 4))
print("\nZ2 forma", Z2.shape, "(1 valor por cliente):", np.round(Z2, 4))
print("\nSalida_Final forma", Salida_Final.shape)
print("Prediccion de la Red (Probabilidad), cliente 1:",
      np.round(Salida_Final[0], 4))
print("Prediccion de la Red (Probabilidad), cliente 2:",
      np.round(Salida_Final[1], 4))

# Verificacion: el lote da lo mismo que procesar cada cliente por separado
for i, cliente in enumerate(X):
    z1 = np.dot(cliente, W1) + b1
    sol = sigmoide(np.dot(sigmoide(z1), W2) + b2)[0]
    assert np.isclose(sol, Salida_Final[i]), "El lote no coincide"
print("\nVerificado: el lote da el mismo resultado que procesar cada "
      "cliente por separado.")

print("""
Por que funciona sin cambiar los pesos:
- np.dot((2x3), (3x4)) -> (2x4): cada FILA de X (un cliente) se
  multiplica por las mismas 4 columnas de W1. La dimension interna (3)
  debe coincidir; la dimension de lote (2) simplemente se arrastra.
- b1 (4,) se "transmite" (broadcasting) y se suma a cada fila de Z1.
- np.dot((2x4), (4,)) -> (2,): una salida por cliente.
- Los pesos describen la red; el lote solo dice CUANTOS ejemplos pasan
  por ella. Con 2 o con 2 millones de clientes las matrices de pesos
  son las mismas. Por eso las GPU son tan utiles: hacen este calculo
  matricial en paralelo (el poder del calculo tensorial).""")
