"""
TALLER DE LABORATORIO: FRONTERAS NO LINEALES
Asignatura: Inteligencia Artificial II - SVM con kernel lineal vs RBF

Mision practica:
  1. Transcribir y ejecutar el codigo base; comparar los vectores de
     soporte con los del taller analitico (Taller 8).
  2. "Enganar" a la frontera lineal agregando X: [5, 5] con etiqueta 0.
  3. Reentrenar el modelo lineal.
  4. Cambiar kernel='linear' por kernel='rbf', reentrenar y predecir.
  5. Reflexion: cuando un kernel lineal fallaria y se necesita RBF?

Genera la figura taller_9_fronteras.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # guarda la figura sin abrir ventana
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles

# =====================================================================
# PARTE 1. CODIGO BASE (transcrito del taller)
# =====================================================================
print("=" * 64)
print("PARTE 1. CODIGO BASE: SVM CON KERNEL LINEAL")
print("=" * 64)

# 1. Crear el dataset (X = Coordenadas, Y = Etiquetas binarias 0 o 1)
X = np.array([[2, 2], [3, 3], [4, 2], [6, 6], [7, 8], [8, 7]])
Y = np.array([0, 0, 0, 1, 1, 1])

# 2. Inicializar SVM con Kernel Lineal
modelo_svm = SVC(kernel='linear')

# 3. Entrenar el modelo (Aprender la ecuacion del hiperplano)
modelo_svm.fit(X, Y)

# 4. Extraer los Vectores de Soporte descubiertos por la IA
vectores = modelo_svm.support_vectors_
print("Los Vectores de Soporte son:\n", vectores)

# 5. Prediccion
nuevo_punto = np.array([[5, 4]])
pred = modelo_svm.predict(nuevo_punto)
print("El punto [5,4] pertenece a la clase:", pred[0])

print("""
Coinciden con los del cuaderno (Taller 8)? SI: (3,3), (4,2) y (6,6).
Son los puntos que tocan el borde del margen; (2,2), (7,8) y (8,7)
no aparecen porque estan lejos de la frontera.""")

# =====================================================================
# PARTES 2 y 3. "ENGANAR" A LA FRONTERA LINEAL CON [5, 5] -> CLASE 0
# =====================================================================
print("\n" + "=" * 64)
print("PARTES 2 y 3. AGREGAR [5, 5] (CLASE 0) Y REENTRENAR (LINEAL)")
print("=" * 64)
X2 = np.vstack([X, [5, 5]])
Y2 = np.append(Y, 0)

lineal2 = SVC(kernel='linear').fit(X2, Y2)
w2, b2 = lineal2.coef_[0], lineal2.intercept_[0]
print("Vectores de soporte:\n", lineal2.support_vectors_)
print(f"Frontera: {w2[0]:.2f}*x + {w2[1]:.2f}*y + ({b2:.2f}) = 0"
      f"   (x + y = {-b2 / w2[0]:.0f})")
print(f"Ancho de la calle: {2 / np.linalg.norm(w2):.3f}  "
      f"(antes: {2 / np.linalg.norm(modelo_svm.coef_[0]):.3f})")
print(f"Precision sobre los datos de entrenamiento: "
      f"{lineal2.score(X2, Y2):.0%}")
print("El punto [5,4] pertenece a la clase:", lineal2.predict(nuevo_punto)[0])
print("""
Observacion: el modelo lineal NO falla. [5,5] cae sobre la diagonal, pero
(5,5) suma 10 y (6,6) suma 12, asi que la recta x + y = 11 aun separa
las dos clases. Lo unico que ocurre es que la calle se ESTRECHA de 4.24
a 1.41 y la frontera queda "forzada" contra [5,5] y [6,6]. Los datos
siguen siendo linealmente separables; la falla real aparece mas abajo,
con datos rodeados por la otra clase (parte 5b).""")

# =====================================================================
# PARTE 4. KERNEL RBF
# =====================================================================
print("\n" + "=" * 64)
print("PARTE 4. CAMBIAR A kernel='rbf', REENTRENAR Y PREDECIR")
print("=" * 64)
modelo_rbf = SVC(kernel='rbf')
modelo_rbf.fit(X2, Y2)
print("Vectores de soporte (RBF):\n", modelo_rbf.support_vectors_)
print(f"Precision sobre los datos de entrenamiento: "
      f"{modelo_rbf.score(X2, Y2):.0%}")
print("El punto [5,4] pertenece a la clase:", modelo_rbf.predict(nuevo_punto)[0])
print("""
Con RBF hay mas vectores de soporte (6 de 7): la frontera ya no es una
recta sino una curva flexible que se adapta a los puntos. Para estos
datos la prediccion de [5,4] coincide con la del kernel lineal.""")

# =====================================================================
# PARTE 5b. DEMOSTRACION: CUANDO EL KERNEL LINEAL SI FALLA (anillos)
# =====================================================================
print("\n" + "=" * 64)
print("PARTE 5b. DATOS EN ANILLOS CONCENTRICOS (clase interna rodeada)")
print("=" * 64)
Xc, yc = make_circles(n_samples=200, factor=0.3, noise=0.08, random_state=0)
lin_c = SVC(kernel='linear').fit(Xc, yc)
rbf_c = SVC(kernel='rbf').fit(Xc, yc)
print(f"Precision kernel lineal: {lin_c.score(Xc, yc):.0%}")
print(f"Precision kernel RBF   : {rbf_c.score(Xc, yc):.0%}")
print("Ninguna recta puede separar un anillo de su centro; RBF proyecta "
      "los datos a una dimension superior donde un plano si los separa.")

# =====================================================================
# PARTE 5. REFLEXION
# =====================================================================
print("\n" + "=" * 64)
print("PARTE 5. REFLEXION: CUANDO SE NECESITA RBF EN EL MUNDO REAL?")
print("=" * 64)
print("""
Un kernel lineal falla cuando una clase esta "rodeada" por otra, o
cuando la frontera de decision real es curva o tiene varias regiones:

- Medicina: un indicador (ej. glucosa o temperatura) es anormal tanto
  si esta muy alto como si esta muy bajo, y los pacientes sanos quedan
  en el rango medio. La clase sana esta "en medio" de la enferma y
  ninguna recta las separa. Igual con celulas malignas segun forma y
  textura.
- Reconocimiento facial: una misma persona cambia con iluminacion,
  angulo, gesto y edad. Sus imagenes forman una region compleja y curva
  del espacio de pixeles, entremezclada con la de otras personas; se
  requiere una frontera no lineal.
- Deteccion de fraude: el comportamiento normal ocupa una region
  compacta (montos y horarios habituales) y el fraude aparece a su
  alrededor en muchas direcciones (montos muy altos, muy bajos,
  horarios raros).
- Sensores e industria: fallas que aparecen cuando una variable sale
  del rango normal tanto por arriba como por abajo.

En cambio, si las clases se separan con una recta (como en el Taller
8), el kernel lineal es mas simple, mas rapido y menos propenso al
sobreajuste; usar RBF alli solo agrega complejidad.
""")

# =====================================================================
# FIGURA
# =====================================================================


def frontera(ax, modelo, X_, y_, xlim, ylim, titulo, soportes=True):
    xx, yy = np.meshgrid(np.linspace(*xlim, 300), np.linspace(*ylim, 300))
    z = modelo.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, z > 0, levels=[-0.5, 0.5, 1.5],
                colors=["#cfe0ff", "#ffd9cf"], alpha=0.7)
    ax.contour(xx, yy, z, levels=[-1, 0, 1], colors="k",
               linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
    ax.scatter(X_[y_ == 0, 0], X_[y_ == 0, 1], s=70, facecolors="none",
               edgecolors="blue", linewidths=2, label="Clase 0 (A)")
    ax.scatter(X_[y_ == 1, 0], X_[y_ == 1, 1], s=70, marker="x",
               color="green", linewidths=2, label="Clase 1 (B)")
    if soportes:
        sv = modelo.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=260, facecolors="none",
                   edgecolors="red", linewidths=2, label="Vectores de soporte")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(titulo, fontsize=10)
    ax.grid(True, alpha=0.3)


fig, axs = plt.subplots(2, 2, figsize=(11, 10))
lim = (0, 10)
frontera(axs[0, 0], modelo_svm, X, Y, lim, lim,
         "1. Kernel lineal (dataset original)")
frontera(axs[0, 1], lineal2, X2, Y2, lim, lim,
         "3. Kernel lineal + punto [5,5] (calle estrecha)")
axs[0, 0].scatter(*nuevo_punto[0], marker="*", s=200, color="purple",
                  label="Nuevo [5,4]")
axs[0, 1].scatter(*nuevo_punto[0], marker="*", s=200, color="purple")
frontera(axs[1, 0], modelo_rbf, X2, Y2, lim, lim,
         "4. Kernel RBF + punto [5,5]")
frontera(axs[1, 1], rbf_c, Xc, yc, (-1.5, 1.5), (-1.5, 1.5),
         f"5b. Anillos: RBF {rbf_c.score(Xc, yc):.0%} vs "
         f"lineal {lin_c.score(Xc, yc):.0%}", soportes=False)
axs[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle("Taller 9: Fronteras no lineales (SVM)", fontsize=13)
fig.tight_layout()
fig.savefig("taller_9_fronteras.png", dpi=130)
print("Figura guardada en taller_9_fronteras.png")
