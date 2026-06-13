# Reporte comparativo PRO con Optuna: Naive vs ARIMA

## 1. Enfoque de optimización

Se utilizó Optuna para automatizar la optimización de ambos enfoques. En Naive, Optuna eligió la mejor regla entre rezagos y medias móviles. En ARIMA, Optuna buscó la mejor combinación de p, d y q dentro de un espacio restringido y con penalización por inestabilidad, usando como función objetivo una combinación de MAE y volatilidad de predicciones.

## 2. Mejoras metodológicas incorporadas

El pipeline incorpora: corrección del backtest ARIMA con historia real, transformación logarítmica opcional (USE_LOG_TRANSFORM=False), filtrado de configuraciones explosivas, penalización por inestabilidad y un score final que combina validation, backtest y consistencia.

## 3. Sobre la interpretabilidad / 'feature importance'

Naive y ARIMA no generan feature importance tradicional. Por ello se construyó una interpretabilidad proxy: para Naive se cuantifica cuánto mejora cada regla respecto al baseline naive_lag_1; para ARIMA se reporta el peso relativo de los coeficientes finales y el efecto promedio de los hiperparámetros p, d y q sobre el MAE.

## 4. Resumen de modelos base

| dataset    | model_family   | model        |      mae |     rmse |     mape |
|:-----------|:---------------|:-------------|---------:|---------:|---------:|
| backtest   | arima          | ARIMA(1,1,1) |  37.9255 |  44.1071 |  6.49885 |
| backtest   | naive          | naive_lag_1  |  67.6667 |  72.2684 | 12.0782  |
| validation | arima          | ARIMA(1,1,1) | 114.182  | 140.839  | 19.7452  |
| validation | naive          | naive_lag_1  |  56      |  66.737  |  9.91782 |


## 5. Resumen de modelos optimizados

| dataset    | model_family   | model          |      mae |     rmse |     mape |
|:-----------|:---------------|:---------------|---------:|---------:|---------:|
| backtest   | naive          | naive_lag_2    |  55.3333 |  55.6761 | 10.8732  |
| backtest   | naive          | naive_ma_2     |  39.8333 |  46.6856 |  7.48952 |
| backtest   | naive          | naive_ma_3     |  40.8333 |  46.9243 |  7.85239 |
| backtest   | naive          | naive_wma_3    |  39.9167 |  45.8928 |  7.40241 |
| validation | arima          | ARIMA(0, 0, 0) | 198.559  | 215.2    | 34.053   |
| validation | arima          | ARIMA(0, 0, 1) | 181.264  | 193.479  | 29.2605  |
| validation | arima          | ARIMA(0, 0, 2) | 123.283  | 138.489  | 19.9143  |
| validation | arima          | ARIMA(0, 1, 0) |  95.4    |  98.8767 | 19.1486  |
| validation | arima          | ARIMA(0, 1, 1) |  74.9479 |  80.585  | 14.0208  |
| validation | arima          | ARIMA(0, 1, 2) |  92.2249 |  97.4947 | 18.5971  |
| validation | arima          | ARIMA(1, 0, 0) |  74.7364 |  94.0452 | 13.0008  |
| validation | arima          | ARIMA(1, 0, 1) |  39.6066 |  45.1589 |  6.93398 |
| validation | arima          | ARIMA(1, 0, 2) |  69.0694 |  71.5134 | 11.4208  |
| validation | arima          | ARIMA(1, 1, 0) |  88.4191 |  95.917  | 15.8183  |
| validation | arima          | ARIMA(1, 1, 2) |  97.5814 | 107.109  | 19.2977  |
| validation | arima          | ARIMA(2, 0, 0) |  58.597  |  75.5655 |  9.32835 |
| validation | arima          | ARIMA(2, 0, 1) | 117.52   | 154.871  | 18.9723  |
| validation | arima          | ARIMA(2, 0, 2) | 102.487  | 146.597  | 17.5705  |
| validation | arima          | ARIMA(2, 1, 0) | 116.294  | 137.62   | 22.8775  |
| validation | arima          | ARIMA(2, 1, 2) |  99.6535 | 115.427  | 17.4943  |
| validation | naive          | naive_lag_2    |  87.8    | 100.683  | 16.0027  |
| validation | naive          | naive_ma_2     |  60.7    |  68.6865 | 11.0606  |
| validation | naive          | naive_ma_3     |  77.6444 |  85.1016 | 14.126   |
| validation | naive          | naive_wma_3    |  67.9667 |  74.1338 | 12.3284  |


## 6. Modelos seleccionados

| dataset    | model_family   | model          |     mae |    rmse |     mape |
|:-----------|:---------------|:---------------|--------:|--------:|---------:|
| backtest   | arima          | ARIMA(2, 0, 0) | 16.5761 | 20.1648 |  2.20915 |
| backtest   | naive          | naive_lag_1    | 30      | 31.0483 |  7.23241 |
| backtest   | naive          | naive_ma_2     | 69.5    | 81.7267 | 13.5038  |
| validation | arima          | ARIMA(2, 0, 0) | 49.2658 | 62.5719 |  7.43568 |
| validation | naive          | naive_lag_1    | 70.6    | 79.0886 | 14.7121  |
| validation | naive          | naive_ma_2     | 20.2    | 26.3306 |  3.24153 |


## 7. Aporte proxy de variables naive

| model       |   proxy_importance_share_pct |   improvement_pct_vs_baseline |
|:------------|-----------------------------:|------------------------------:|
| naive_ma_2  |                     26.1332  |                      15.9516  |
| naive_wma_3 |                     22.8462  |                       9.20519 |
| naive_ma_3  |                     13.2385  |                      -3.27937 |
| naive_lag_2 |                      4.44874 |                     -28.5151  |
| naive_lag_1 |                      0       |                       0       |


## 8. Aporte proxy de términos ARIMA

| term   | component_type   |   proxy_importance_share_pct |   abs_coefficient |
|:-------|:-----------------|-----------------------------:|------------------:|
| sigma2 | variance         |                  66.1698     |       4432.12     |
| const  | constant         |                  50.725      |      14861.8      |
| ma.L2  | moving_average   |                   0.008175   |          0.389554 |
| ma.L1  | moving_average   |                   0.00813705 |          0.518999 |
| ar.L1  | autoregressive   |                   0.00682875 |          0.532026 |
| ar.L2  | autoregressive   |                   0.00242094 |          0.769777 |

