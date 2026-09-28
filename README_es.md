*Lee esto en otros idiomas: [English](README.md)*

# Proyecto de Deep Learning — Mantenimiento Predictivo para Motores Turbofán (NASA C-MAPSS)

## Descripción General

Este proyecto desarrolla un avanzado **benchmark de Deep Learning para Mantenimiento Predictivo** utilizando el conjunto de datos **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**.

En lugar de predecir la **Vida Útil Remanente (RUL, Remaining Useful Life)** exacta mediante regresión, el problema se reformula como una **tarea de clasificación multiclase de series temporales multivariadas**, donde el estado de salud del motor se categoriza en distintos niveles de riesgo operativo.

La solución evalúa y compara **cinco arquitecturas de Deep Learning de última generación** para clasificación de series temporales multivariadas, implementadas en **PyTorch** siguiendo un flujo de trabajo integral de Machine Learning industrial.

---

# Acerca del Dataset

**Dataset:** Conjunto de Datos de Simulación de Degradación de Motores Turbofán (C-MAPSS)

**Fuente:** NASA Prognostics Data Repository

---

# Descripción General

El dataset C-MAPSS simula el proceso de degradación de motores turbofán de aeronaves que operan bajo diferentes condiciones ambientales y modos de falla.

Cada motor es monitoreado mediante múltiples canales de sensores que capturan su comportamiento termodinámico a lo largo del tiempo hasta que ocurre la falla.

El conjunto de datos contiene:

* Configuraciones operativas
* Mediciones de sensores
* Identificadores de motores
* Trayectorias completas de degradación

---

# Reformulación del Problema

En lugar de predecir la Vida Útil Remanente (RUL) exacta, el problema se transforma en una tarea de clasificación multiclase.

Los estados de salud del motor se definen de la siguiente manera:

| Clase                  | Vida Útil Remanente   |
| ---------------------- | --------------------- |
| Saludable (Healthy)    | RUL > 60 ciclos       |
| Alerta (Alert)         | 30 < RUL ≤ 60 ciclos  |
| Crítico (Critical)     | RUL ≤ 30 ciclos       |

Esta formulación proporciona resultados más accionables para la planificación de mantenimiento y la toma de decisiones industriales.

---

# Ingeniería de Características

El proyecto incluye amplias técnicas de ingeniería de características diseñadas para mejorar la detección de patrones de degradación.

* **Características dinámicas:** media y desviación estándar móviles causales (ventana = 5 ciclos) de cada sensor, calculadas por motor.
* **Características estáticas:** firma base del motor (media de cada sensor en los primeros 5 ciclos) y un embedding del sub-dataset (FD001–FD004, es decir, condiciones operativas / modos de falla).
* **Preparación de datos:** escalado Z-score ajustado solo con la partición de entrenamiento y ventanas deslizantes de 30 ciclos (tensores `[Ventanas, 30, 66]`), etiquetadas con la clase del último ciclo de cada ventana.

---

# Prevención de Data Leakage

Un aspecto crítico de los proyectos de mantenimiento predictivo es evitar la fuga de información.

Para garantizar una evaluación realista:

* Las trayectorias completas de cada motor permanecen dentro de una única partición.
* Particionamiento Train/Validation/Test basado en grupos (70% / 15% / 15% de los motores, `GroupShuffleSplit`).
* El escalador y la detección de sensores invariantes se ajustan solo con la partición de entrenamiento.
* No se utiliza información futura durante la generación de características.
* Los valores de RUL se eliminan después de la construcción de la variable objetivo.

---

# Arquitecturas de Deep Learning Evaluadas

Se implementaron y compararon cinco arquitecturas avanzadas.

## 1. FCN (Fully Convolutional Network)

Línea base tradicional y sólida para clasificación de series temporales.

---

## 2. InceptionTime

Arquitectura convolucional de última generación para clasificación de series temporales.

---

## 3. Time Series Transformer

Arquitectura basada en Transformers que utiliza mecanismos de autoatención.

---

## 4. PatchTST

Arquitectura Transformer reciente de última generación.

---

## 5. ConvTransformer

Arquitectura híbrida CNN-Transformer.

---

# Estrategia de Entrenamiento

La canalización de entrenamiento incluye:

* Implementación en PyTorch
* Aceleración mediante GPU
* Early Stopping
* Gradient Clipping
* Balanceo de clases mediante pesos
* Optimizador Adam
* Función de pérdida CrossEntropy
* Semillas aleatorias fijas para reproducibilidad

**Selección de modelo:** la mejor arquitectura se elige por **Macro F1 en validación**; la partición de test solo se usa para reportar las métricas finales.

**Datos de evaluación:** todas las particiones se construyen a partir de las trayectorias completas hasta la falla `train_FD00x`, ya que las etiquetas requieren conocer el RUL de cada ciclo. Los archivos oficiales `test_FD00x` / `RUL_FD00x` se incluyen en `data/` como referencia.

---

# Resultados Generados

El proyecto produce automáticamente:

* Tabla comparativa de benchmark
* Comparación de Accuracy
* Comparación de Macro F1
* Comparación de tiempos de entrenamiento
* Matriz de confusión
* Informe de clasificación
* Curvas ROC
* Visualizaciones de degradación de motores
* Identificación del mejor modelo

---

# Resultados

Métricas sobre el set de test (107 motores no vistos, 20.641 ventanas). Entrenamiento en una NVIDIA GeForce GTX 1650.

| Modelo                | Macro F1 Val | Macro F1 Test | Accuracy Test | Tiempo de entrenamiento (s) |
| --------------------- | :----------: | :-----------: | :-----------: | :-------------------------: |
| **PatchTST**          | **0.779**    | 0.792         | **0.841**     | 504                         |
| TimeSeriesTransformer | 0.774        | **0.794**     | 0.834         | 221                         |
| ConvTransformer       | 0.750        | 0.738         | 0.792         | 213                         |
| InceptionTime         | 0.723        | 0.753         | 0.803         | 104                         |
| FCN (Baseline)        | 0.683        | 0.681         | 0.750         | 134                         |

**Modelo seleccionado: PatchTST** (mejor Macro F1 en validación). En test alcanza un ROC AUC de 0.988 (Crítico), 0.967 (Saludable) y 0.895 (Alerta). La clase *Alerta* es la más difícil (F1 = 0.61) porque es la zona de transición entre la operación normal y la falla inminente, mientras que los motores en estado *Crítico* se detectan con un recall de 0.83 y casi nunca se confunden con *Saludable* (5 de 3.317 ventanas).

Ambos modelos basados en Transformers superan claramente a las líneas base convolucionales; TimeSeriesTransformer ofrece un Macro F1 similar en menos de la mitad del tiempo de entrenamiento de PatchTST.

---

# Estructura del Proyecto

```
├── data/
│   ├── train_FD001.txt ... train_FD004.txt   # trayectorias hasta la falla (utilizadas)
│   ├── test_FD001.txt  ... test_FD004.txt    # set de test oficial de la NASA (referencia)
│   ├── RUL_FD001.txt   ... RUL_FD004.txt     # RUL oficial del test (referencia)
│   ├── readme.txt                            # descripción del dataset (NASA)
│   └── Damage Propagation Modeling.pdf       # paper de referencia (Saxena et al., 2008)
├── notebooks/
│   └── Predictive Maintenance.ipynb          # pipeline completo: EDA → FE → 5 modelos → benchmark
├── requirements.txt
├── README.md
└── README_es.md
```

---

# Cómo Ejecutarlo

```bash
git clone https://github.com/arguar13/Predictive-Maintenance-for-Turbofan-Eng-Time-Series-Deep-Learning-Classification-.git
cd Predictive-Maintenance-for-Turbofan-Eng-Time-Series-Deep-Learning-Classification-
pip install -r requirements.txt
jupyter notebook "notebooks/Predictive Maintenance.ipynb"
```

El notebook lee los datos desde la carpeta relativa `data/` y usa la GPU automáticamente si está disponible (también funciona en CPU).

---

# Impacto de Negocio

Esta solución puede apoyar:

* Planificación de mantenimiento predictivo
* Gestión de flotas aeronáuticas
* Planificación de repuestos
* Reducción de tiempos de inactividad
* Prevención de fallas
* Monitoreo de la salud de los activos

El enfoque multiclase proporciona alertas de mantenimiento interpretables y directamente utilizables por los equipos operativos.

---

# Licencia

Este proyecto está destinado a fines educativos, de investigación y de portafolio.

Dataset proporcionado por el Centro de Investigación NASA Ames.

---

## Autor

**Armando Guarnera**
Data Scientist
Argentina
