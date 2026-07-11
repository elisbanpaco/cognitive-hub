<div align="center">
  <img src="https://img.shields.io/badge/Course_Nivel-Intermedio-F37626?style=for-the-badge" alt="Nivel Intermedio" />
  <h1>📚 Aprendizaje No Supervisado y Reducción de Dimensiones</h1>
  <p><strong>De los datos crudos a los patrones ocultos: Un enfoque práctico e intuitivo.</strong></p>
</div>

---

¡Hola y bienvenido a este módulo de Machine Learning! 👋🏼 

A diferencia del aprendizaje supervisado, donde le damos a nuestro modelo las respuestas correctas por adelantado, en el **Aprendizaje No Supervisado** somos verdaderos exploradores. Aquí, los datos no tienen etiquetas. Nuestro objetivo es dejar que los algoritmos descubran la estructura subyacente, encuentren grupos naturales y simplifiquen la información sin perder su esencia.

Si alguna vez te has preguntado cómo Netflix agrupa perfiles de usuarios similares o cómo los biólogos clasifican secuencias genéticas, estás en el lugar correcto.

## 🎯 ¿Qué logramos en este módulo? (Resultados Prácticos)

En este laboratorio, hemos implementado y documentado flujos de trabajo analíticos de principio a fin. Concretamente, en esta investigación se logró:
- **Agrupación Jerárquica de Datos:** Modelamos la similitud matemática entre observaciones sin requerir etiquetas previas, construyendo dendrogramas que revelan la jerarquía natural de los conjuntos de datos.
- **Reducción de Dimensionalidad Espacial:** Comprimimos datos altamente complejos (con docenas de variables) a 2 o 3 componentes principales mediante PCA, logrando retener la varianza original para facilitar su visualización e interpretación.
- **Extracción de Características (Factorización):** Aplicamos matrices no negativas (NMF) para descomponer datos no estructurados en sus partes más puras y fundamentales.
- **Ingeniería Analítica en Python:** Estructuramos todo el pipeline de manera reproducible, utilizando buenas prácticas de código apoyándonos en el ecosistema de **Scikit-Learn, Pandas y NumPy**.

## 🗂️ Tu Plan de Estudio (Los Cuadernos)

Hemos diseñado este repositorio con una filosofía educativa. Cada Notebook es una "lección" completa con teoría explicada de forma sencilla, código ejecutable, e interpretaciones paso a paso.

| Módulo | Tema Principal | Algoritmos que Dominarás | ¿Para qué sirve en la vida real? |
|:---|:---|:---|:---|
| 📓 [**Módulo 1:**<br>Clustering Jerárquico](./clustering_jerarquico.ipynb) | Descubrir familias en los datos construyendo árboles de decisión (Dendrogramas). | *Clustering Aglomerativo* | Segmentar clientes en marketing según sus hábitos de compra, creando nichos específicos. |
| 📓 [**Módulo 2:**<br>Análisis de Componentes Principales](./transformacion_PCA.ipynb) | Reducir el "ruido" y las columnas redundantes de tus datos maximizando la varianza. | *PCA (Principal Component Analysis)* | Comprimir datos financieros masivos para visualizarlos en 2D o 3D sin perder las tendencias. |
| 📓 [**Módulo 3:**<br>Factorización No Negativa](./factorizacion_matriz_no_negativa_NMF.ipynb) | Descomponer datos complejos en sus partes más básicas y fundamentales. | *NMF (Non-Negative Matrix Factorization)* | Extraer temas principales de miles de artículos de noticias simultáneamente. |

## ⚙️ Configura tu Entorno de Trabajo (Laboratorio)

Para que puedas ejecutar los cuadernos en tu computadora exactamente como lo haríamos en un laboratorio, utilizamos [`uv`](https://github.com/astral-sh/uv), una herramienta ultrarrápida que instala todo lo que necesitas en segundos.

Sigue estos pasos en tu terminal:

**1. Descarga los materiales:**
```bash
git clone https://github.com/elisbanpaco/cognitive-hub.git
cd cognitive-hub/ml_research/Aprendizaje-no-supervisado-reduccion-dimensiones
```

**2. Instala las librerías necesarias:**
```bash
# uv leerá el archivo pyproject.toml e instalará mágicamente Pandas, Scikit-Learn, Matplotlib, etc.
uv sync
```

**3. Activa tu entorno de laboratorio:**
- **En Mac/Linux:** `source .venv/bin/activate`
- **En Windows:** `.venv\Scripts\activate`

**4. ¡Abre Jupyter y empieza a aprender!**
```bash
jupyter notebook
```

## 🧠 Prerrequisitos

Para aprovechar al máximo este módulo sin frustrarte, te recomendamos estar familiarizado con:
- Sintaxis básica de **Python**.
- Manipulación de tablas con **Pandas** y operaciones matemáticas con **NumPy**.
- Conceptos básicos de álgebra lineal (como qué es una matriz y un vector).

## 🏆 Proyecto Abierto (Práctica Extra)

El aprendizaje nunca se detiene. Si completaste los cuadernos y quieres un desafío extra, ¡te invitamos a colaborar en este repositorio! 

¿Por qué no intentas investigar e implementar otros algoritmos como **K-Means**, **DBSCAN** o **t-SNE**?
Haz un *fork* del repositorio, crea tu propio Notebook con el mismo estilo explicativo y envíanos un *Pull Request*. Estaremos encantados de revisar tu código y agregarlo al plan de estudio.

¡Feliz aprendizaje y que disfrutes descubriendo patrones! 🚀
