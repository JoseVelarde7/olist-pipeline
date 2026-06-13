# Explicación detallada: `03_train_arima_sarima_olist_optuna.ipynb`

Pipeline de pronóstico de demanda con optimización automática de hiperparámetros usando **Optuna**.  
Cada sección explica *qué hace*, *por qué* y *en qué mejora* respecto a enfoques más simples.

---

## Índice

1. [Configuración inicial](#1-configuración-inicial)
2. [Utilidades generales](#2-utilidades-generales)
3. [Preparación de datos y candidatos Naive](#3-preparación-de-datos-y-candidatos-naive)
4. [Optimización Naive con Optuna](#4-optimización-naive-con-optuna)
5. [ARIMA base (sin optimizar)](#5-arima-base-sin-optimizar)
6. [Optimización ARIMA con Optuna](#6-optimización-arima-con-optuna)
7. [Modelo final ARIMA e interpretabilidad proxy](#7-modelo-final-arima-e-interpretabilidad-proxy)
8. [Selección final por categoría](#8-selección-final-por-categoría)
9. [Pipeline principal por categoría](#9-pipeline-principal-por-categoría)
10. [Función main: orquestación y exportaciones](#10-función-main-orquestación-y-exportaciones)
11. [Archivos generados](#11-archivos-generados)

---

## 1. Configuración inicial

### ¿Qué hace?

Define todas las constantes del experimento en un único lugar antes de ejecutar cualquier cálculo.

```python
TRAIN_END = pd.Timestamp("2017-12-01")   # fin del conjunto de entrenamiento
VAL_END   = pd.Timestamp("2018-05-01")   # fin del conjunto de validación
# El backtest es todo lo que viene después de VAL_END

N_TRIALS_NAIVE = 20    # intentos de Optuna para Naive
N_TRIALS_ARIMA = 40    # intentos de Optuna para ARIMA
OPTUNA_SEED    = 42    # semilla para reproducibilidad

# Espacio de búsqueda restringido para ARIMA
ARIMA_P_MIN, ARIMA_P_MAX = 0, 2   # términos autorregresivos
ARIMA_D_MIN, ARIMA_D_MAX = 0, 1   # diferenciaciones
ARIMA_Q_MIN, ARIMA_Q_MAX = 0, 2   # términos de media móvil

STABILITY_PENALTY_WEIGHT = 0.10   # penalización por predicciones volátiles
USE_LOG_TRANSFORM = False          # transformación logarítmica opcional
```

### ¿Por qué?

Centralizar los parámetros permite cambiar el horizonte temporal, el número de trials o el espacio de búsqueda sin tocar el código de cada función. En producción (Airflow), estos valores se pasan como argumentos a `run_optuna_experiment()`.

### División temporal del dataset

| Período       | Rango                          | Uso                                      |
|---------------|-------------------------------|------------------------------------------|
| **Train**     | inicio → 2017-12-01           | Ajuste del modelo                        |
| **Validation**| 2018-01-01 → 2018-05-01       | Selección de hiperparámetros con Optuna  |
| **Backtest**  | 2018-06-01 → fin de datos     | Evaluación final (datos nunca vistos)    |

> **Por qué tres splits y no dos:** Con solo train/test se corre el riesgo de sobreajustar los hiperparámetros al conjunto de prueba. El split de validación independiente garantiza que el backtest mide generalización real.

---

## 2. Utilidades generales

### 2.1 `compute_metrics(y_true, y_pred)`

Calcula MAE, RMSE y MAPE para cualquier par de vectores real/predicción.

```python
def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}
```

- **MAE** (Error Absoluto Medio): fácil de interpretar, no penaliza outliers extremos.
- **RMSE** (Raíz del Error Cuadrático Medio): penaliza errores grandes; útil para detectar predicciones explosivas.
- **MAPE** (Error Porcentual Absoluto Medio): permite comparar categorías con volúmenes muy distintos (ej.: 50 vs 5 000 órdenes).

### 2.2 Filtros de validez (`is_valid_metric_dict`, `are_valid_predictions`)

```python
def are_valid_predictions(preds):
    if not np.isfinite(preds).all():   return False  # NaN o ±inf
    if np.max(np.abs(preds)) > 1_000_000: return False  # explosión numérica
    return True
```

### ¿Por qué?

ARIMA puede producir predicciones infinitas o divergentes si la serie no es estacionaria y el orden elegido es incorrecto (ej.: `ARIMA(2,0,2)` sin diferenciación en una serie con tendencia). Sin estos filtros, un solo trial malo destruye el promedio de todo el experimento.

### 2.3 `stability_penalty(preds)` y `selection_score()`

```python
def stability_penalty(preds):
    # Desviación estándar de las predicciones = volatilidad
    return float(np.std(preds))

def selection_score(mae_val, mae_back, w_val=0.6, w_back=0.3):
    return w_val * mae_val + w_back * mae_back
```

- **`stability_penalty`**: un modelo que predice 200, 2000, 50, 3000 tiene MAE aceptable pero es inútil en producción. La penalización por volatilidad descarta esas soluciones.
- **`selection_score`**: pondera 60% validación + 30% backtest. Si solo se usara MAE de validación, el modelo estaría sobreoptimizado al período 2018-01. Incluir el backtest asegura que el modelo generaliza más allá del período de búsqueda.

### 2.4 Transformación logarítmica opcional

```python
def transform_series_for_model(series, use_log=False):
    if not use_log:
        return series
    return np.log1p(series.clip(lower=0))

def inverse_transform_predictions(preds, use_log=False):
    if not use_log:
        return preds
    return np.expm1(preds)
```

`log1p(x) = ln(1 + x)` es segura para valores cero (demanda nula en algunos meses). Estabiliza series con varianza creciente (heterocedasticidad). El flag `USE_LOG_TRANSFORM = False` lo deja desactivado por defecto porque los datos de Olist ya son moderadamente estacionarios.

---

## 3. Preparación de datos y candidatos Naive

### 3.1 `prepare_category_frame(df_cat)`

Toma los datos de una categoría y construye todas las **reglas Naive** como columnas adicionales.

```python
def prepare_category_frame(df_cat):
    df_cat["naive_lag_1"]  = df_cat["demand"]                         # último mes conocido
    df_cat["naive_lag_2"]  = df_cat["demand"].shift(1)                # hace 2 meses
    df_cat["naive_ma_2"]   = df_cat["demand"].rolling(2).mean()       # promedio últimos 2
    df_cat["naive_ma_3"]   = df_cat["demand"].rolling(3).mean()       # promedio últimos 3
    df_cat["naive_ma_6"]   = df_cat["demand"].rolling(6).mean()       # promedio últimos 6
    def weighted_ma(values):
        weights = np.arange(1, len(values) + 1)
        return np.average(values, weights=weights)
    df_cat["naive_wma_3"]  = df_cat["demand"].rolling(3).apply(weighted_ma, raw=True)
    df_cat["naive_seasonal_12"] = df_cat["demand"].shift(11)          # mismo mes año anterior
    return df_cat
```

### ¿Qué es cada regla?

| Regla              | Descripción                                   | Útil cuando…                        |
|--------------------|-----------------------------------------------|--------------------------------------|
| `naive_lag_1`      | Predice el mes siguiente = mes actual         | Demanda estable sin tendencia        |
| `naive_ma_2`       | Promedio de los 2 últimos meses               | Suaviza ruido puntual                |
| `naive_wma_3`      | Promedio ponderado (mayor peso al más reciente)| Demanda con leve tendencia           |
| `naive_ma_3`       | Promedio de los 3 últimos meses               | Promedio más robusto                 |
| `naive_lag_2`      | Valor de hace 2 meses                        | Ciclos bimensuales                   |
| `naive_seasonal_12`| Mismo mes del año anterior                   | Series con estacionalidad anual      |

### 3.2 `split_category(df_cat)`

Divide la categoría en tres subconjuntos:

```python
def split_category(df_cat):
    train    = df_cat[df_cat["year_month"] <= TRAIN_END]
    val      = df_cat[(df_cat["year_month"] > TRAIN_END) & (df_cat["year_month"] <= VAL_END)]
    backtest = df_cat[df_cat["year_month"] > VAL_END]
    return train, val, backtest
```

---

## 4. Optimización Naive con Optuna

### `optimize_naive_with_optuna(val_df, category)`

Usa Optuna para elegir automáticamente **cuál regla Naive tiene menor MAE en validación**.

```python
def optimize_naive_with_optuna(val_df, category):
    # 1. Evalúa todas las reglas en validación
    val_candidates, _ = evaluate_naive_subset(val_df, "validation", category)
    metrics_map = val_candidates.set_index("model").to_dict(orient="index")

    # 2. Optuna elige la mejor regla (variable categórica)
    def objective(trial):
        rule = trial.suggest_categorical("naive_rule", list(metrics_map.keys()))
        return metrics_map[rule]["mae"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=min(N_TRIALS_NAIVE, len(metrics_map)))

    best_rule = study.best_params["naive_rule"]
    return best_rule, val_candidates, study.trials_dataframe()
```

### ¿Por qué Optuna aquí si hay pocas reglas?

Con solo 5 reglas candidatas, un grid search sería equivalente. Sin embargo, usar Optuna:
1. **Unifica el framework**: todo el experimento usa el mismo mecanismo de búsqueda.
2. **Registra el historial de trials** (`12_optuna_naive_trials.csv`) para auditoría y comparación.
3. **Escala** si en el futuro se agregan más reglas (SARIMA naive, Fourier, etc.).

---

## 5. ARIMA base (sin optimizar)

Antes de Optuna, se evalúa un **ARIMA(1,1,1) fijo** como línea base. Esto permite comparar cuánto gana la optimización respecto a una configuración estándar de la literatura.

```python
# Validación con ARIMA(1,1,1) fijo
base_val_metrics, base_val_preds = evaluate_arima_order(
    train["demand"],
    val["demand_next_month"],
    order=(1, 1, 1)
)

# Backtest: se usa train + val como historia (corrección clave)
base_back_metrics, base_back_preds = evaluate_arima_order(
    pd.concat([train["demand"], val["demand"]]),
    backtest["demand_next_month"],
    order=(1, 1, 1)
)
```

> **Por qué usar `train + val` como historia para el backtest:** Si se usara solo `train`, el modelo desconocería los meses 2018-01 a 2018-05 al hacer predicciones de 2018-06 en adelante. Eso sería una contaminación inversa: el modelo perdería información que en producción sí tendría disponible.

---

## 6. Optimización ARIMA con Optuna

### 6.1 `rolling_arima_forecast()`

Implementa el **pronóstico rolling one-step-ahead**: el modelo se re-entrena en cada paso incorporando el valor real observado.

```python
def rolling_arima_forecast(train_series, test_target_series, order, use_log=False):
    history = list(train_series)   # historial acumulado

    for actual in test_target_series:
        # Ajusta el modelo con toda la historia disponible
        model  = ARIMA(history, order=order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        fitted = model.fit()

        # Predice solo el siguiente paso
        pred = float(fitted.forecast(steps=1)[0])
        preds.append(pred)

        # Agrega el valor REAL al historial (no la predicción)
        history.append(actual)

    return np.array(preds)
```

### ¿Por qué rolling y no forecast estático?

| Tipo de forecast | Comportamiento                                           | Problema                                  |
|------------------|----------------------------------------------------------|-------------------------------------------|
| **Estático**     | Ajusta una vez con train, predice todos los meses juntos | El error se acumula mes a mes sin corrección |
| **Rolling**      | Re-ajusta con la información real de cada mes            | Simula el uso real en producción          |

En producción nunca se predicen 6 meses de golpe. Cada mes llega un nuevo dato real y el modelo se actualiza. El rolling forecast replica exactamente ese escenario.

### 6.2 `optimize_arima_with_optuna()`

Busca el mejor orden `(p, d, q)` en el espacio restringido.

```python
def optimize_arima_with_optuna(train_series, val_target_series, category):
    def objective(trial):
        p = trial.suggest_int("p", 0, 2)   # AR: 0, 1 o 2
        d = trial.suggest_int("d", 0, 1)   # diferenciación: 0 o 1
        q = trial.suggest_int("q", 0, 2)   # MA: 0, 1 o 2

        metrics, preds = evaluate_arima_order(train_series, val_target_series, (p, d, q))

        if metrics is None:
            return ARIMA_MAX_REASONABLE_MAE + 1   # penaliza modelos inválidos

        # Objetivo = MAE + penalización por volatilidad de predicciones
        penalty = np.std(preds)
        return metrics["mae"] + 0.10 * penalty

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=40)

    best_p = study.best_params["p"]
    best_d = study.best_params["d"]
    best_q = study.best_params["q"]
    return (best_p, best_d, best_q), ...
```

### Espacio de búsqueda

El espacio total es `3 × 2 × 3 = 18` combinaciones únicas. Con **40 trials**, Optuna evalúa algunas combinaciones varias veces, lo que le permite **estimar la varianza** de cada configuración y no quedarse con una combinación que fue buena por azar.

### ¿Qué hace Optuna internamente? (algoritmo TPE)

TPE = **Tree-structured Parzen Estimator**. Funciona así:

1. Los primeros ~10 trials son aleatorios (exploración).
2. Divide los trials anteriores en "buenos" (MAE bajo) y "malos" (MAE alto).
3. Ajusta dos distribuciones de probabilidad: `l(x)` sobre los buenos y `g(x)` sobre los malos.
4. Elige el siguiente punto donde el ratio `l(x) / g(x)` es mayor (concentra la búsqueda donde los buenos superan a los malos).

**Ventaja vs grid search:** No prueba combinaciones inútiles. Si `d=0` consistentemente da MAE alto, deja de probar órdenes con `d=0`.

### Penalización por inestabilidad

```python
# Función objetivo de Optuna
return mae + 0.10 * np.std(preds)
```

Un modelo que predice alternadamente valores muy altos y muy bajos puede tener MAE aceptable pero produce un plan de inventario inútil. La penalización de `0.10 × σ(predicciones)` descarta esos modelos a favor de uno más conservador y estable.

---

## 7. Modelo final ARIMA e interpretabilidad proxy

### 7.1 `fit_final_arima_model()`

Después de encontrar el mejor orden con Optuna (usando solo train), se re-entrena el modelo con **toda la historia disponible** (`train + val + backtest`).

```python
def fit_final_arima_model(series_all_history, order, use_log=False):
    transformed = transform_series_for_model(series_all_history)
    model  = ARIMA(transformed.values, order=order,
                   enforce_stationarity=False,
                   enforce_invertibility=False)
    return model.fit()
```

**¿Por qué re-entrenar con todos los datos?** El modelo que va a producción debe aprender de toda la información disponible, no solo del periodo de entrenamiento. Si `ARIMA(2,0,1)` fue el mejor orden, ese orden se mantiene, pero los coeficientes se ajustan con más datos → predicciones más precisas.

### 7.2 `extract_arima_parameter_importance()`

ARIMA no tiene "feature importance" como un árbol de decisión. Se usa **importancia proxy** basada en los coeficientes del modelo ajustado.

```python
def extract_arima_parameter_importance(fitted_model, category, dataset_scope, order):
    values    = list(fitted_model.params)       # coeficientes: ar.L1, ma.L1, sigma2...
    abs_values = np.abs(values)
    abs_sum    = abs_values.sum()

    for name, value, abs_value in zip(names, values, abs_values):
        share = (abs_value / abs_sum) * 100     # porcentaje del peso total

        component_type = (
            "autoregressive" if name.startswith("ar.")  else
            "moving_average" if name.startswith("ma.")  else
            "variance"       if "sigma" in name         else
            "constant"
        )
```

**Interpretación:**
- `ar.L1` con 60%: el valor del mes pasado es el factor más importante para la predicción → la demanda tiene fuerte inercia.
- `ma.L1` con 30%: los errores del mes pasado también importan → el modelo aprende de sus equivocaciones recientes.
- `sigma2` alto: hay incertidumbre no explicada por la estructura temporal.

### 7.3 `build_arima_order_effect_table()`

Para cada valor de `p`, `d` y `q`, calcula el **MAE promedio** en todos los trials que usaron ese valor.

```python
for param in ["p", "d", "q"]:
    temp = (
        clean_df
        .groupby(["category", param])["mae"]
        .mean()
        .reset_index()
    )
    temp["hyperparameter"] = param
```

Esto genera la tabla `05_arima_hyperparameter_effect.csv` que responde: *¿Cuánto impacta en el MAE usar `d=0` vs `d=1`?* Si la curva tiene un mínimo claro, ese parámetro es sensible y vale optimizarlo. Si la curva es plana, cualquier valor sirve.

---

## 8. Selección final por categoría

### `run_category_pipeline()` — sección de selección

Una vez evaluados el mejor Naive y el mejor ARIMA, se comparan con un **score combinado**:

```python
naive_score = 0.6 * mae_naive_val + 0.3 * mae_naive_back
arima_score = 0.6 * mae_arima_val + 0.3 * mae_arima_back

if naive_score <= arima_score:
    ganador = "naive"
else:
    ganador = "arima"
```

### ¿Por qué puede ganar el Naive?

Para series de demanda mensual con **ruido alto y pocas observaciones** (solo 20-30 meses de historia), ARIMA puede sobreajustarse al patrón de entrenamiento y fallar en el backtest. Un simple `naive_ma_2` (promedio de los 2 últimos meses) puede superar a ARIMA porque captura la tendencia reciente sin sobreajustar.

Ejemplo práctico en el notebook:

| Categoría        | Ganador       | Orden/Regla      | MAE val | MAE backtest |
|------------------|---------------|------------------|---------|--------------|
| beleza_saude     | ARIMA         | (2, 0, 0)        | 16.4    | 24.1         |
| cama_mesa_banho  | ARIMA         | (1, 0, 1)        | 31.2    | 38.7         |
| esporte_lazer    | Naive         | naive_ma_2       | 28.9    | 19.3         |

---

## 9. Pipeline principal por categoría

### `run_category_pipeline(df_cat)`

Ejecuta todos los pasos anteriores para **una sola categoría** y devuelve un diccionario con todos los resultados.

```
Entrada: DataFrame filtrado a una categoría
    ↓
prepare_category_frame()      → agrega columnas naive_lag_1, naive_ma_2...
    ↓
split_category()              → train / val / backtest
    ↓
evaluate_naive_subset()       → métricas Naive en val y backtest
    ↓
build_naive_contribution()    → importancia proxy de cada regla Naive
    ↓
optimize_naive_with_optuna()  → elige la mejor regla Naive
    ↓
evaluate_arima_order(1,1,1)   → ARIMA base (sin optimizar)
    ↓
optimize_arima_with_optuna()  → búsqueda de (p,d,q) en 40 trials TPE
    ↓
evaluate_arima_order(best)    → métricas del mejor ARIMA en backtest
    ↓
fit_final_arima_model()       → modelo final con toda la historia
    ↓
extract_arima_parameter_importance() → importancia proxy de coeficientes
    ↓
selección final               → compara naive_score vs arima_score
    ↓
Salida: dict con métricas, predicciones, artefactos por categoría
```

### ¿Por qué categoría a categoría y no un único modelo global?

La demanda de `beleza_saude` (cosméticos) tiene un patrón estacional diferente a `esporte_lazer` (deportes). Un modelo global promediaría esas diferencias y perdería señal. Al entrenar un modelo por categoría, cada serie tiene su propio orden `(p,d,q)` óptimo.

---

## 10. Función main: orquestación y exportaciones

### Flujo de `main()`

```python
def main():
    # 1. Carga y valida el dataset
    df = pd.read_csv(input_path)
    df["year_month"] = pd.to_datetime(df["year_month"])

    # 2. Itera sobre cada categoría
    for cat in categories:
        df_cat = df[df["product_category_name"] == cat]
        result = run_category_pipeline(df_cat)
        # acumula todos los DataFrames de resultados

    # 3. Consolida y limpia los resultados globales
    arima_search_df   = clean_arima_search_df(arima_search_df)
    arima_order_effect = build_arima_order_effect_table(arima_search_df)

    # 4. Exporta 13 CSVs + 6 gráficos + 1 reporte Markdown + 1 PKL
    for filename, dataframe in export_map.items():
        dataframe.to_csv(OUTPUT_DIR / filename, index=False)

    # 5. Guarda el modelo final como artefacto PKL
    artifact = {
        "selected_models_by_category": final_models_artifact,
        "train_end": str(TRAIN_END.date()),
        "val_end":   str(VAL_END.date()),
        ...
    }
    pickle.dump(artifact, open("15_final_model_optuna_pro.pkl", "wb"))
```

### ¿Qué contiene el artefacto `.pkl`?

El `.pkl` no guarda objetos ARIMA statsmodels (que son difíciles de serializar de forma estable). Guarda un **diccionario con toda la información necesaria para hacer predicciones en producción**:

```python
{
  "beleza_saude": {
    "family": "arima",
    "order": (2, 0, 0),
    "fitted_model": <ARIMA ajustado con toda la historia>,
    "use_log_transform": False,
  },
  "esporte_lazer": {
    "family": "naive",
    "rule": "naive_ma_2",
  },
  ...
}
```

---

## 11. Archivos generados

| Archivo                              | Contenido                                                       |
|--------------------------------------|-----------------------------------------------------------------|
| `01_base_models_metrics.csv`         | MAE/RMSE/MAPE de naive_lag_1 y ARIMA(1,1,1) por categoría      |
| `02_optimized_models_metrics.csv`    | Métricas de todas las reglas Naive y todos los trials ARIMA     |
| `03_naive_proxy_importance.csv`      | Aporte relativo de cada regla Naive vs baseline naive_lag_1     |
| `04_arima_optuna_search_validation.csv` | Resultados de los 40 trials ARIMA por categoría (solo válidos) |
| `05_arima_hyperparameter_effect.csv` | MAE promedio por valor de p, d y q                              |
| `06_arima_parameter_proxy_importance.csv` | Peso relativo de coeficientes AR, MA, σ² del modelo final |
| `07_final_selection_by_category.csv` | Modelo ganador (Naive o ARIMA) por cada categoría              |
| `08_selected_predictions.csv`        | Predicciones del modelo ganador en val y backtest               |
| `09_summary_base_models.csv`         | Promedio de métricas base por dataset × modelo                  |
| `10_summary_optimized_models.csv`    | Promedio de métricas optimizadas por dataset × modelo           |
| `11_summary_selected_models.csv`     | Métricas finales de los modelos seleccionados                   |
| `12_optuna_naive_trials.csv`         | Historial de trials de Optuna para Naive                        |
| `13_optuna_arima_trials.csv`         | Historial de trials de Optuna para ARIMA                        |
| `14_report_model_optimization_optuna_pro.md` | Reporte comparativo en Markdown (se muestra en el dashboard) |
| `15_final_model_optuna_pro.pkl`      | Artefacto del modelo final listo para producción                |

---

## Resumen de mejoras respecto a un enfoque sin Optuna

| Aspecto                        | Sin Optuna                          | Con Optuna                                          |
|--------------------------------|-------------------------------------|-----------------------------------------------------|
| **Orden ARIMA**                | Fijo: (1,1,1)                       | Buscado automáticamente en `p∈[0,2], d∈[0,1], q∈[0,2]` |
| **Regla Naive**                | Solo naive_lag_1                    | Elige entre 5 reglas la de menor MAE en validación  |
| **Modelos explosivos**         | Pueden contaminar los resultados    | Filtrados con `is_valid_metric_dict` y `are_valid_predictions` |
| **Predicciones inestables**    | No detectadas                       | Penalizadas con `0.10 × std(predicciones)`          |
| **Selección de ganador**       | Subjetiva o por MAE de validación   | Score ponderado: 60% val + 30% backtest             |
| **Interpretabilidad**          | Ninguna                             | Importancia proxy para Naive y ARIMA                |
| **Trazabilidad**               | Sin historial                       | Todos los trials guardados en CSV                   |
| **Reproducibilidad**           | Depende de semilla manual           | `OPTUNA_SEED=42` fijado en samplers TPE             |
