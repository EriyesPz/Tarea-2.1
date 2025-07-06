# Tarea: Regresión Lineal

En clase, al analizar el conjunto de datos de casas, utilizamos regresión lineal para predecir el precio de las viviendas. Sin embargo, algunos datos presentan límites fijos, por ejemplo:

- Si una casa tiene **50 años o más**, el dato indica **50**.
- Si el **precio** de una casa es **500,000 o más**, el dato indica **500,000**.
- **Ingresos** con un máximo de **15**.

El problema con estos límites es que el modelo puede tener dificultades para ajustarse, ya que algunos datos están restringidos y otros no (como el tamaño, número de cuartos, etc.).

## Instrucciones

1. **Carga y procesamiento de datos:**  
    Deben cargar los datos y procesarlos de manera que el **porcentaje de precisión de entrenamiento (score) alcance al menos un 90%** (deseable), sin sesgar el modelo.  
    Puedes eliminar columnas o registros, agregar nuevas características, etc., para lograr este valor.

2. **Separación de datos:**  
    Realiza la separación en datos de entrenamiento y prueba con el conjunto de datos resultante.

3. **Análisis de resultados:**  
    - ¿El resultado fue mejor o peor?
    - ¿Por qué crees que es así?

4. **Entrega:**  
    Deben entregar un archivo de Colab descargado o compartido de manera pública, o bien, un repositorio con el análisis de los datos.  
    Además, incluye tu análisis a las dos preguntas anteriores.  
    Los cambios aplicados al DataFrame deben estar sustentados y explicados en comentarios.
