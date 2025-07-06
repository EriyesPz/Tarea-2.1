# Tarea-2.1 — Regresión Lineal en el Set de Casas

**Alumno:** Erick Josué Reyes Paz  
**Cuenta:** 20222001353


## Estructura del repositorio

```
Tarea-2.1/
├── docs/
│   ├── instrucciones.md
│   └── respuesta.md
├── data/
│   └── housing.csv 
├── src/
│   ├── modelos.py            # Modelo de regresión lineal (pipeline)
│   └── procesos.py           # Procesamiento y limpieza del DataFrame
├── main.py                   # Script principal de entrenamiento y evaluación
├── README.md                
└── RESPUESTAS.md            
```

### Librerías necesarias

- pandas
- numpy
- scikit-learn
- seaborn

### Instalación de dependencias

1. **Crea un entorno virtual** (opcional pero recomendado):

    ```bash
    python -m venv venv
    # En Windows
    venv\Scripts\activate
    # En Linux/Mac
    source venv/bin/activate
    ```

2. **Instala las librerías necesarias:**

    ```bash
    pip install pandas numpy scikit-learn
    ```

## Ejecución del proyecto

1. Desde la raíz del proyecto, ejecuta:

    ```bash
    python main.py
    ```

El programa mostrará en consola:

- Score de entrenamiento
- Score de prueba
- RMSE (Error cuadrático medio)

## Respuestas y análisis

Las respuestas al enunciado y el análisis del rendimiento del modelo se encuentran en el archivo [`docs/respuesta.md`](docs/respuesta.md). Ahí se explican:
- ¿El resultado fue mejor o peor?
- ¿Por qué crees que es así?