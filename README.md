# 🧠 CognitiveHub

![Python](https://img.shields.io/badge/python-3.14-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**CognitiveHub** es un ecosistema integral y monorepo diseñado para la investigación, experimentación y despliegue en producción de modelos de Machine Learning y Deep Learning, con un enfoque en neurociencia cognitiva.

## ✨ Características Principales

- **Arquitectura Desacoplada:** Separación clara entre el entorno de entrenamiento (ML) y el despliegue en producción (API + Cliente).
- **Inferencia Rápida:** API construida con FastAPI para servir modelos de forma asíncrona.
- **Interfaz Moderna:** Frontend reactivo construido con Next.js y Tailwind CSS.

---

## 🏗️ Estructura del Proyecto

```
CognitiveHub/
├── client/          # 💻 Frontend (Next.js 16 + React 19 + TypeScript)
├── api/             # ⚙️ Backend de Inferencia (FastAPI + Python)
└── ml_research/     # 🔬 Laboratorio ML (Jupyter, PyTorch/TensorFlow)
```

## 🚀 Guía de Inicio Rápido (Quick Start)

### 0. Prerrequisitos

Asegúrate de tener instalado: Node.js (v20+), pnpm (v9+), Python (3.12+) y Git.

### 1. Variables de Entorno (Importante)

Copia los archivos de ejemplo y configura tus variables locales. **NUNCA subas tus archivos .env reales al repositorio.**

```bash
cp api/.env.example api/.env
cp client/.env.example client/.env
```

### 2. Levantar los Servicios Locales

**Frontend (Cliente):**

```bash
cd client
pnpm install
pnpm dev
```

**Backend (API):**

```bash
cd api
python -m venv venv
# Activar: 'source venv/bin/activate' (Mac/Linux) o 'venv\Scripts\activate' (Windows)
pip install -r requirements.txt
uvicorn main:app --reload
```

**Investigación (ML Research):**

```bash
cd ml_research
python -m venv venv
# Activar entorno virtual
pip install -r requirements.txt
jupyter lab
```

## 🔌 Puertos de Desarrollo

| Servicio | URL Local | Descripción |
|----------|----------|------------|
| Frontend | http://localhost:3000 | Interfaz de usuario |
| API | http://localhost:8000/docs | Swagger UI interactivo de la API |
| Jupyter | http://localhost:8888 | Entorno de experimentación ML |

---

## 🤝 Guía para Contribuir

¡Las contribuciones son bienvenidas! Para mantener la consistencia en el equipo, sigue estos pasos:

1. Haz un Fork del repositorio y clónalo localmente.
2. Crea una rama descriptiva: `git checkout -b feature/nueva-funcion` o `fix/error-login`.
3. Haz commits siguiendo la convención de Conventional Commits:
   - `feat: agrega red neuronal para predicción`
   - `fix: resuelve problema de CORS en FastAPI`
4. Sube los cambios: `git push origin feature/nueva-funcion`.
5. Abre un Pull Request (PR) hacia la rama main y espera el Code Review.

## ⚠️ Reglas Estrictas del Repositorio

- **No commitear:** `.env`, `node_modules/`, `venv/`.
- Los modelos pesados (.pt, .h5) y los datasets grandes deben ignorarse. Usa Git LFS o descárgalos externamente.
- El Code Review es obligatorio antes de realizar un merge.

## 📄 Licencia

Distribuido bajo la Licencia MIT.