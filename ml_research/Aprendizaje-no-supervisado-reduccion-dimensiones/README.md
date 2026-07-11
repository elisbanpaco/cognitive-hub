<div align="center">
  <h1>🧠 Aprendizaje No Supervisado y Reducción de Dimensiones</h1>
  <p>
    <strong>Investigación en Machine Learning — Cognitive Hub</strong>
  </p>

  [![Python Version](https://img.shields.io/badge/python-%3E%3D3.13-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
  [![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)](https://jupyter.org/)
  [![uv](https://img.shields.io/badge/uv-Fast%20Python%20Package%20Manager-purple.svg?style=for-the-badge)](https://github.com/astral-sh/uv)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
</div>

---

Bienvenido al módulo de **Aprendizaje No Supervisado y Reducción de Dimensiones**. Este proyecto forma parte del ecosistema de investigación del **Cognitive Hub**, enfocado en la exploración matemática y algorítmica de conjuntos de datos no etiquetados. Aquí aplicamos técnicas de vanguardia para descubrir patrones ocultos, extraer características esenciales y reducir la dimensionalidad de los datos.

## 📑 Índice

- [Objetivos del Proyecto](#-objetivos-del-proyecto)
- [Arquitectura del Repositorio](#-arquitectura-del-repositorio)
- [Guía de Inicio Rápido](#-guía-de-inicio-rápido)
- [Stack Tecnológico](#-stack-tecnológico)
- [Cómo Contribuir](#-cómo-contribuir)
- [Licencia](#-licencia)

## 📌 Objetivos del Proyecto

Nuestro enfoque se centra en tres pilares fundamentales:
1. **Descubrimiento de Conocimiento (Clustering):** Agrupar datos complejos basándonos en sus similitudes intrínsecas, sin depender de etiquetas predefinidas.
2. **Reducción de Dimensionalidad:** Mitigar la *maldición de la dimensionalidad* extrayendo los componentes que capturan la mayor varianza, facilitando el procesamiento y reduciendo el costo computacional.
3. **Interpretación y Visualización:** Proyectar espacios multidimensionales en 2D y 3D para la toma de decisiones informada.

## 📂 Arquitectura del Repositorio

El flujo de trabajo e investigación está estructurado en una serie de Notebooks iterativos. Cada uno contiene teoría, implementación matemática y visualizaciones.

| Notebook | Descripción | Algoritmos Clave |
| :--- | :--- | :--- |
| 📓 [`clustering-jerarquico.ipynb`](./clustering-jerarquico.ipynb) | Análisis de similitud y jerarquías en los datos. Incluye la generación e interpretación de dendrogramas. | *Clustering Aglomerativo, Enlaces (Ward, Complete, Average)* |
| 📓 [`transformacion-PCA.ipynb`](./transformacion-PCA.ipynb) | Compresión de características y reducción espacial. Evaluación de la varianza explicada y visualización de componentes principales. | *PCA, SVD (Descomposición en Valores Singulares)* |
| 📓 [`factorizacion_matriz_no_negativa_NMF.ipynb`](./factorizacion_matriz_no_negativa_NMF.ipynb) | Extracción de partes constituyentes de los datos. Ideal para procesamiento de imágenes, señales y modelado de temas en NLP. | *NMF (Non-Negative Matrix Factorization)* |

## 🚀 Guía de Inicio Rápido

El proyecto está diseñado para ser reproducible y rápido de configurar utilizando [`uv`](https://github.com/astral-sh/uv), el gestor de paquetes de nueva generación para Python.

### Requisitos Previos
- Python 3.13 o superior.
- Git.

### Instalación Paso a Paso

**1. Clonar el repositorio localmente:**
```bash
git clone https://github.com/elisbanpaco/cognitive-hub.git
cd cognitive-hub/ml_research/Aprendizaje-no-supervisado-reduccion-dimensiones
```

**2. Sincronizar el entorno:**
Este comando creará automáticamente un entorno virtual aislado (`.venv`) e instalará las versiones exactas estipuladas en el archivo `uv.lock`.
```bash
uv sync
```

**3. Activar el entorno virtual:**
- **Sistemas Unix (Linux/macOS):**
  ```bash
  source .venv/bin/activate
  ```
- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```

**4. Levantar el entorno de Jupyter:**
```bash
jupyter notebook
```

## 🛠️ Stack Tecnológico

El proyecto se apoya en un ecosistema robusto de ciencia de datos:

- **[Scikit-Learn (>=1.9.0)](https://scikit-learn.org/):** El núcleo de nuestros modelos de Machine Learning.
- **[NumPy (>=2.5.1)](https://numpy.org/) & [Pandas (>=3.0.3)](https://pandas.pydata.org/):** Para manipulación de matrices, tensores y estructuras de datos.
- **[Matplotlib (>=3.11.0)](https://matplotlib.org/):** Para renderizado de proyecciones y gráficos de alta calidad.
- **Gestión de Entorno:** [uv](https://docs.astral.sh/uv/) y configuración centralizada vía `pyproject.toml`.

## 🤝 Cómo Contribuir

Fomentamos la colaboración y el crecimiento continuo. Si tienes ideas para nuevos algoritmos (ej. *t-SNE*, *UMAP*, *DBSCAN*), o quieres optimizar los actuales:

1. Realiza un **Fork** de este repositorio.
2. Crea una rama descriptiva: `git checkout -b feature/nombre-algoritmo`.
3. Documenta tu código y asegúrate de que los notebooks incluyan celdas explicativas.
4. Confirma tus cambios: `git commit -m 'feat: Añadida implementación de t-SNE'`.
5. Haz push a tu fork: `git push origin feature/nombre-algoritmo`.
6. Abre un **Pull Request** para iniciar el proceso de revisión.

## 📜 Licencia

Este proyecto se distribuye bajo la **Licencia MIT**. Siéntete libre de utilizar, modificar y distribuir el código con fines académicos o comerciales. Consulta el archivo `LICENSE` para más detalles.

---
<div align="center">
  <sub>Desarrollado con ❤️ por la comunidad del Cognitive Hub.</sub>
</div>
