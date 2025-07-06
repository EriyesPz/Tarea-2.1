# 📝 Análisis de Resultados — Tarea de Regresión Lineal

## 1. ¿El resultado fue mejor o peor?

El resultado fue muy favorable, considerando las limitaciones estructurales del conjunto de datos. El modelo ajustado mostró un desempeño sólido, reflejando una relación coherente entre las variables predictoras y el valor de la vivienda.

Durante el proceso, se logró capturar patrones representativos del comportamiento del mercado inmobiliario con un enfoque completamente basado en regresión lineal, manteniendo un balance adecuado entre complejidad y generalización. Esto evidencia una buena capacidad del modelo para predecir valores fuera del conjunto de entrenamiento.

## 2. ¿Por qué crees que es así?

- **Procesamiento de datos enfocado en relevancia y estabilidad:**  
    Se diseñaron nuevas variables (features derivadas) que representan mejor la estructura del problema, como la relación entre cuartos y hogares (`rooms_per_household`), o la proporción de población (`population_per_household`). También se crearon interacciones útiles, como `income * age`, y se aplicaron transformaciones logarítmicas para estabilizar la varianza.

- **Reducción de sesgos por truncamiento:**  
    Se manejaron adecuadamente los topes artificiales del dataset, como el valor máximo de las casas o los ingresos, evitando que esos límites distorsionaran las relaciones aprendidas por el modelo.


- **Métrica de desempeño confiable:**  
    El modelo final mostró una capacidad predictiva confiable y estable, con una métrica de error (RMSE) ajustada y una explicación estadística clara (R alto en ambos conjuntos).
