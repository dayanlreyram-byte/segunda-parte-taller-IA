"""
TALLER ANALITICO: DIBUJANDO EL MARGEN
Asignatura: Inteligencia Artificial II - Sesion 10: SVM (Support Vector Machine)

Puntos en el plano (ejes X e Y de 0 a 10):
    Clase A (circulos): (2,2), (3,3), (4,2)
    Clase B (equis):    (6,6), (7,8), (8,7)

  1. Trazar la recta optima con el mismo espacio de "calle" a ambos lados.
  2. Marcar en rojo los Vectores de Soporte.
  3. Si se agrega un punto de Clase A en (1,1), cambia la linea?

Este script resuelve el taller con un SVM lineal de margen duro (C muy
grande) y genera el dibujo en taller_8_margen.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # guarda la figura sin abrir ventana
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# ---------------------------------------------------------------- datos
A = np.array([[2, 2], [3, 3], [4, 2]])   # Clase A (-1)
B = np.array([[6, 6], [7, 8], [8, 7]])   # Clase B (+1)
X = np.vstack([A, B])
y = np.array([-1, -1, -1, 1, 1, 1])


def entrenar(X, y):
    """SVM lineal de margen duro: C enorme = no se permiten errores."""
    modelo = SVC(kernel="linear", C=1e6)
    modelo.fit(X, y)
    w = modelo.coef_[0]
    b = modelo.intercept_[0]
    margen = 2 / np.linalg.norm(w)      # ancho total de la calle
    return modelo, w, b, margen


modelo, w, b, margen = entrenar(X, y)

# ------------------------------------------------- resultados (puntos 1 y 2)
print("=" * 62)
print("1. RECTA OPTIMA (hiperplano de maximo margen)")
print("=" * 62)
print(f"w = {np.round(w, 4)},  b = {b:.4f}")
print(f"Ecuacion: {w[0]:.3f}*x + {w[1]:.3f}*y + ({b:.3f}) = 0")
print("Simplificada: x + y = 9   (y = 9 - x)")
print(f"Ancho total de la calle (margen) = 2/||w|| = {margen:.4f}"
      f"  (= 3*raiz(2))")
print(f"Espacio a cada lado = {margen / 2:.4f}  (= 3/raiz(2))")

print("\n" + "=" * 62)
print("2. VECTORES DE SOPORTE")
print("=" * 62)
soportes = modelo.support_vectors_
for p in soportes:
    clase = "A" if tuple(p) in {tuple(a) for a in A} else "B"
    dist = abs(w @ p + b) / np.linalg.norm(w)
    print(f"  {tuple(int(v) for v in p)}  Clase {clase}  "
          f"distancia a la recta = {dist:.4f}")
print("Los demas puntos, (2,2), (7,8) y (8,7), estan mas lejos: no "
      "sostienen la frontera.")

# ------------------------------------------------------------- punto 3
print("\n" + "=" * 62)
print("3. AGREGAR UN PUNTO DE CLASE A EN (1,1)")
print("=" * 62)
X2 = np.vstack([X, [1, 1]])
y2 = np.append(y, -1)
modelo2, w2, b2, margen2 = entrenar(X2, y2)
misma = np.allclose(w, w2, atol=1e-3) and np.isclose(b, b2, atol=1e-3)
print(f"Antes : w = {np.round(w, 4)}, b = {b:.4f}, margen = {margen:.4f}")
print(f"Despues: w = {np.round(w2, 4)}, b = {b2:.4f}, margen = {margen2:.4f}")
print("Vectores de soporte despues:",
      [tuple(int(v) for v in p) for p in modelo2.support_vectors_])
print(">> La linea CAMBIA?", "NO" if misma else "SI")
print("""
Justificacion (teoria de SVM):
- SVM solo depende de los Vectores de Soporte, los puntos criticos que
  tocan el borde del margen. Los puntos lejanos de la frontera no
  influyen en la posicion del hiperplano.
- (1,1) queda detras de (2,2), mucho mas lejos de la calle que (3,3) y
  (4,2). No entra al margen ni es soporte, asi que la recta x + y = 9
  y el ancho de la calle se mantienen iguales.
- Contraste con KNN: alli todos los datos se memorizan y votan; en SVM
  se pueden borrar los puntos que no son soporte y el modelo no cambia.
""")

# ------------------------------------------------------------------ dibujo
fig, ax = plt.subplots(figsize=(7, 7))
xs = np.linspace(0, 10, 100)


def y_de(x, c):
    """Recta w0*x + w1*y + b = c  ->  y."""
    return (c - b - w[0] * x) / w[1]


ax.plot(xs, y_de(xs, 0), "k-", lw=2, label="Recta optima  x + y = 9")
ax.plot(xs, y_de(xs, 1), "k--", lw=1, label="Bordes del margen")
ax.plot(xs, y_de(xs, -1), "k--", lw=1)
ax.fill_between(xs, y_de(xs, -1), y_de(xs, 1), color="gold", alpha=0.25,
                label=f"Calle (ancho = {margen:.2f})")

ax.scatter(A[:, 0], A[:, 1], s=110, facecolors="none", edgecolors="blue",
           linewidths=2, label="Clase A (circulos)")
ax.scatter(B[:, 0], B[:, 1], s=110, marker="x", color="green", linewidths=2,
           label="Clase B (equis)")
ax.scatter(soportes[:, 0], soportes[:, 1], s=420, facecolors="none",
           edgecolors="red", linewidths=2.5, label="Vectores de Soporte")

for x0, y0 in X:
    ax.annotate(f"({x0},{y0})", (x0, y0), textcoords="offset points",
                xytext=(8, 8), fontsize=9)

# punto hipotetico del punto 3
ax.scatter([1], [1], s=110, facecolors="none", edgecolors="gray",
           linewidths=1.5, linestyle="--")
ax.annotate("(1,1) nuevo A\nno cambia la linea", (1, 1),
            textcoords="offset points", xytext=(8, -28), fontsize=8,
            color="gray")

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xticks(range(11))
ax.set_yticks(range(11))
ax.set_aspect("equal")
ax.grid(True, alpha=0.4)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Taller: Dibujando el margen (SVM)")
ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
fig.tight_layout()
fig.savefig("taller_8_margen.png", dpi=150)
print("Dibujo guardado en taller_8_margen.png")
