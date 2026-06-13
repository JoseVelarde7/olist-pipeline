# Guion de exposición — ~5 minutos
### Sprint 3: Hiperparametrización y modelo final de demanda · Olist

> **Ritmo:** hablar despacio, pausar al cambiar de concepto.
> Las marcas ⏱ son referencias de tiempo aproximado.

---

## DIAPOSITIVA 1 — Introducción ⏱ 0:00 – 1:45

---

Buenas. En este sprint el objetivo fue construir un sistema que **prediga automáticamente cuántos pedidos va a recibir Olist cada mes** por categoría de producto, sin tener que elegir el modelo manualmente.

Para eso evaluamos dos familias de modelos. Los llamo familias porque cada una agrupa varias estrategias distintas.

---

**La primera familia es Naive** — modelos de referencia.

La idea central de Naive es muy simple: *"el futuro se va a parecer al pasado reciente."*

Piensen en la categoría de cosméticos, `beleza_saude`. Supongamos que las ventas de los últimos meses fueron así:

> Enero: 800 pedidos | Febrero: 850 | Marzo: 780 | **¿Abril?**

El modelo **lag_1** —que en español sería rezago 1— dice: *"lo que pasó el mes pasado, va a pasar este mes."* Predicción de abril = 780.

El **lag_2** —rezago 2— se va un paso más atrás: *"me baso en lo de hace dos meses."* Predicción de abril = 850.

¿Por qué "rezago"? Porque el valor viaja en el tiempo, llega tarde, *rezagado*. lag_1 llega con un mes de retraso, lag_2 con dos.

Las **medias móviles** van un poco más lejos: en vez de tomar un solo mes, promedian varios. `ma_2` promedia los últimos dos: (780 + 850) / 2 = 815. `ma_3` promedia los últimos tres. Eso suaviza picos raros que no van a repetirse.

Y `wma_3` es la media móvil **ponderada**: el mes más reciente pesa más que el de hace tres meses. Porque lo de ayer importa más que lo de hace tres meses.

---

**La segunda familia es ARIMA** — modelos estadísticos con tres parámetros: p, d y q.

- **p** es el componente **autorregresivo**. *Auto* significa "a sí mismo": el modelo usa sus propios valores pasados para predecir. Si p=2, mira los dos meses anteriores y estima cuánto influyen en el mes siguiente. Es como decir que la demanda tiene inercia propia.

- **d** es la **diferenciación**. En vez de trabajar con ventas absolutas, trabaja con los *cambios*. Si d=1, no usa 780 y 850, usa "subió 70" y "bajó 70". Eso estabiliza la serie y facilita que el modelo encuentre patrones.

- **q** es la **media móvil del error**. El modelo aprende de sus propios errores: si el mes pasado predijo 800 y la demanda real fue 780, ese error de 20 unidades entra al cálculo del mes siguiente para corregirse solo.

Las tres métricas que ven en la diapositiva —MAE, RMSE y MAPE— las explico en la siguiente.

---

## DIAPOSITIVA 2 — Enfoque de Optimización con Optuna ⏱ 1:45 – 3:30

---

El problema de ARIMA es que p, d y q hay que elegirlos. Si elegimos p entre 0 y 2, d entre 0 y 1, y q entre 0 y 2, tenemos 18 combinaciones posibles. Por categoría. Y a eso se suma elegir cuál regla Naive es mejor.

Hacer eso a mano es lento y sesgado: uno termina eligiendo lo que le va bien al período que ya conoce, no al futuro. Ahí entra **Optuna**.

Optuna es una librería que automatiza esa búsqueda. Lo hace en cuatro pasos —que son los que ven en pantalla:

---

**Paso 1 — Espacio de búsqueda.**
Le decimos a Optuna qué puede probar. Para Naive: las 5 reglas de rezagos y medias. Para ARIMA: los valores de p, d y q dentro de los rangos que definimos.

**Paso 2 — Evaluación por trial.**
En cada intento —llamado *trial*— Optuna toma una configuración, la entrena con los datos hasta diciembre 2017, y mide el error en el período de **validación** de enero a mayo 2018. Aquí entran las tres métricas:

- **MAE** —Error Absoluto Medio—: si el MAE es 30, en promedio nos equivocamos en 30 pedidos por mes. Es la más fácil de interpretar.
- **RMSE** —Raíz del Error Cuadrático—: penaliza más los errores grandes. Si una vez fallamos por 200 pedidos, eso pesa mucho más que fallar 10 veces por 20.
- **MAPE** —Error Porcentual—: el error como porcentaje. Nos permite comparar categorías de volumen muy distinto. Un error de 30 pedidos no es lo mismo si la categoría vende 100 que si vende 5.000.

Optuna busca reducir el MAE en cada trial. No prueba todo al azar: usa un algoritmo llamado TPE que aprende de cada intento. Para entenderlo simple: imaginen que buscan el punto más bajo de un terreno con los ojos vendados. La búsqueda aleatoria toca el suelo en puntos al azar. TPE en cambio *recuerda* dónde encontró tierra baja y el siguiente paso lo da cerca de ahí. En 40 intentos cubre mejor el espacio que un grid search completo.

**Paso 3 — Filtrado de estabilidad.**
No basta con que el error sea bajo. También importa que las predicciones sean estables. Si un modelo predice 200 pedidos en enero, 5.000 en febrero y 100 en marzo, el MAE promedio puede ser aceptable pero ese modelo es inútil para planificar inventario. Optuna penaliza esas configuraciones erráticas.

**Paso 4 — Selección final.**
El ganador no se elige solo por el error en validación. Se combina: 60% el error en validación más 30% el error en **backtest**. El backtest son los meses de junio 2018 en adelante, datos que el modelo nunca vio durante el entrenamiento. Si funciona bien ahí, es que aprendió el patrón real, no que memorizó los datos. Esta ventaja clave está resaltada en la diapositiva: *"reduce el riesgo de elegir modelos que solo funcionan bien en una muestra específica."*

---

## DIAPOSITIVA 3 — Mejoras Metodológicas Incorporadas ⏱ 3:30 – 5:00

---

Esta diapositiva muestra las cuatro correcciones técnicas que hicimos para que los resultados sean reproducibles y confiables, no solo buenos en papel.

---

**Corrección 01 — Corrección del Backtest.**
Cuando el modelo hace predicciones en el período de backtest, necesita conocer toda la historia real hasta ese punto. En una versión anterior, los datos de entrenamiento incluían órdenes sintéticas de septiembre 2018 que no existían en realidad. Eso hacía que agosto 2018 tuviera una demanda proyectada irreal, lo que distorsionaba completamente las métricas. La corrección fue usar únicamente los datos reales del equipo para el entrenamiento.

**Corrección 02 — Filtrado de Inestables.**
Algunas combinaciones de p, d y q hacen que ARIMA produzca predicciones que explotan numéricamente: un mes predice 200, el siguiente 2.000.000. Si no se filtran, arruinan el promedio de todo el experimento. El pipeline descarta automáticamente cualquier configuración cuyas predicciones superen umbrales razonables o contengan valores infinitos.

**Corrección 03 — Penalización de Erráticos.**
Incluso sin valores extremos, un modelo puede ser inestable: oscila mucho entre meses. Se mide eso con la desviación estándar de las predicciones y se suma como penalización al objetivo de Optuna. Así, entre dos modelos con el mismo MAE, gana el más estable.

**Corrección 04 — Evaluación Conjunta.**
En vez de elegir el modelo que mejor funciona en un solo período, se pondera validación y backtest al mismo tiempo. Esto evita que Optuna encuentre un modelo que memorizó los meses de validación pero falla en los meses siguientes. En nuestro caso, para `esporte_lazer` el modelo ARIMA ganaba en validación pero el Naive ganaba al considerar también el backtest, y ese fue el resultado final correcto.

---

Para cerrar: el sistema completo corre automáticamente desde Airflow, genera todos los archivos de resultados, y los muestra en el dashboard en tiempo real. Cada vez que el pipeline corre, los gráficos y métricas se actualizan solos.

---

*Fin del guion — tiempo aproximado: 5 minutos.*
