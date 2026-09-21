"""
TALLER DE LABORATORIO: HACKEANDO LOS PESOS
Asignatura: Inteligencia Artificial II - El Perceptron

Mision practica: la Compuerta OR
  1. Transcribir el codigo del perceptron.
  2. Verificar la compuerta AND con [1,0], [0,1] y [0,0] (todas dan 0).
  3. El reto: modificar MANUALMENTE los pesos (W) y/o el sesgo (b) para
     que la neurona resuelva la compuerta OR.
  4. Reglas del OR: 1 si las entradas son [1,1], [1,0] o [0,1]; 0 solo
     si la entrada es [0,0].
  5. Anotar los pesos y el sesgo que resuelven el problema.
"""
import itertools
import numpy as np

# =====================================================================
# PARTE 1. CODIGO DEL PERCEPTRON (transcrito del taller)
# =====================================================================


# 1. Definir la Funcion de Activacion (Escalon)
def funcion_escalon(z):
    if z >= 0:
        return 1
    else:
        return 0


# 2. Definir la Estructura de la Neurona
def perceptron(X, W, b):
    # Producto punto (Combinacion lineal)
    # Equivalente a: (X[0]*W[0]) + (X[1]*W[1]) ...
    Z = np.dot(X, W) + b

    # Activacion
    salida = funcion_escalon(Z)
    return salida


# 3. Datos del problema (Compuerta Logica AND)
# El AND solo da 1 si ambas entradas son 1.
entradas = np.array([1, 1])   # Vector X
pesos = np.array([0.5, 0.5])  # Vector W
sesgo = -0.8                  # Constante b

# 4. Inferencia (Forward pass)
resultado = perceptron(entradas, pesos, sesgo)
print("El Perceptron disparo el valor:", resultado)

# =====================================================================
# PARTE 2. VERIFICAR LA COMPUERTA AND (tabla de verdad completa)
# =====================================================================
TODAS = [np.array(e) for e in ([0, 0], [0, 1], [1, 0], [1, 1])]
AND = [0, 0, 0, 1]
OR = [0, 1, 1, 1]


def tabla(nombre, W, b, esperado):
    """Imprime la tabla de verdad de la neurona y devuelve si es correcta."""
    print(f"\nCompuerta {nombre}:  W = {np.asarray(W).tolist()},  b = {b}")
    print(f"  {'X':<8}{'Z = X.W + b':>14}{'salida':>9}{'esperado':>10}")
    ok = True
    for x, esp in zip(TODAS, esperado):
        z = np.dot(x, W) + b
        s = perceptron(x, W, b)
        ok &= (s == esp)
        print(f"  {str(x.tolist()):<8}{z:>14.2f}{s:>9}{esp:>10}"
              f"   {'OK' if s == esp else 'ERROR'}")
    print(f"  >> {'Resuelve' if ok else 'NO resuelve'} la compuerta {nombre}")
    return ok


print("\n" + "=" * 64)
print("PARTE 2. VERIFICACION DE LA COMPUERTA AND (W = [0.5, 0.5], b = -0.8)")
print("=" * 64)
tabla("AND", pesos, sesgo, AND)
print("\nLas entradas [1,0], [0,1] y [0,0] dan Z < 0, asi que arrojan 0.")

# =====================================================================
# PARTE 3. EL RETO: HACKEAR LOS PESOS PARA LA COMPUERTA OR
# =====================================================================
print("\n" + "=" * 64)
print("PARTE 3. EL RETO: COMPUERTA OR")
print("=" * 64)
print("""
Razonamiento: la neurona dispara cuando Z = w1*x1 + w2*x2 + b >= 0.
Con las cuatro entradas del OR se obtienen cuatro desigualdades:

   [0,0] -> 0 :  b            <  0     (no debe disparar)
   [1,0] -> 1 :  w1 + b       >= 0     (debe disparar)
   [0,1] -> 1 :  w2 + b       >= 0     (debe disparar)
   [1,1] -> 1 :  w1 + w2 + b  >= 0     (se cumple sola si las dos
                                        anteriores se cumplen)

Basta con que UNA sola entrada activa alcance para superar el umbral.
Con el AND, en cambio, el sesgo (-0.8) era tan negativo que hacian
falta las DOS entradas. Para el OR se sube el sesgo (menos negativo) o
se suben los pesos.

Solucion elegida: pesos W = [1.0, 1.0] y sesgo b = -0.5.
   [0,0] : 0 + 0 - 0.5 = -0.5  -> 0
   [1,0] : 1 + 0 - 0.5 =  0.5  -> 1
   [0,1] : 0 + 1 - 0.5 =  0.5  -> 1
   [1,1] : 1 + 1 - 0.5 =  1.5  -> 1""")

pesos_or = np.array([1.0, 1.0])
sesgo_or = -0.5
ok_or = tabla("OR", pesos_or, sesgo_or, OR)
assert ok_or

# Otra solucion valida, cambiando solo el sesgo del codigo original
print("\nTambien funciona dejando los pesos originales y cambiando SOLO "
      "el sesgo:")
assert tabla("OR", pesos, -0.3, OR)

# =====================================================================
# PARTE 4. NO HAY UNA SOLA SOLUCION: BUSQUEDA EXHAUSTIVA
# =====================================================================
print("\n" + "=" * 64)
print("PARTE 4. CUANTAS COMBINACIONES RESUELVEN EL OR?")
print("=" * 64)
rejilla = np.round(np.arange(-1.0, 1.01, 0.1), 1)
soluciones = [(float(w1), float(w2), float(b))
              for w1, w2, b in itertools.product(rejilla, repeat=3)
              if all(perceptron(x, np.array([w1, w2]), b) == e
                     for x, e in zip(TODAS, OR))]
print(f"Se probaron {len(rejilla) ** 3} combinaciones (w1, w2, b) en "
      f"[-1, 1] con paso 0.1.")
print(f"Resuelven el OR: {len(soluciones)} combinaciones.")
print("Ejemplos:", soluciones[:3], "...")

# Bonus: el XOR no tiene solucion con una sola neurona
XOR = [0, 1, 1, 0]
sol_xor = [(w1, w2, b) for w1, w2, b in itertools.product(rejilla, repeat=3)
           if all(perceptron(x, np.array([w1, w2]), b) == e
                  for x, e in zip(TODAS, XOR))]
print(f"\nBonus: combinaciones que resuelven el XOR: {len(sol_xor)}")
print("""Ninguna. Un solo perceptron traza UNA recta (w1*x1 + w2*x2 + b = 0)
y el XOR no se puede separar con una recta. Por eso se necesitan capas
ocultas, tema de la red neuronal multicapa.""")

# =====================================================================
# PARTE 5. RESPUESTA FINAL
# =====================================================================
print("\n" + "=" * 64)
print("PARTE 5. PESOS Y SESGO QUE RESUELVEN LA COMPUERTA OR")
print("=" * 64)
print(f"  pesos = np.array({pesos_or.tolist()})")
print(f"  sesgo = {sesgo_or}")
print("""
En la proxima clase la red neuronal hara este ajuste sola: el
entrenamiento consiste en encontrar automaticamente los pesos y el
sesgo, en lugar de "hackearlos" a mano.""")
