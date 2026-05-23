"""
Лабораторная работа №5
Методы: GammaRegressor, OrthogonalMatchingPursuitCV
Датасет: California Housing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import GammaRegressor, OrthogonalMatchingPursuitCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 1. Загрузка данных (регрессионный датасет)
housing = fetch_california_housing()
X, y = housing.data, housing.target

print("Размерность матрицы признаков:", X.shape)
print("Уникальные классы:", np.unique(y)[:5], "...")  
print("Целевая переменная (медианная стоимость дома):\n", pd.Series(y).describe())

# 2. Целевой столбец и признаки
# Целевой столбец: y (медианная стоимость дома в квартале, в сотнях тысяч долларов)
# Признаки: 8 числовых показателей (средний доход, возраст домов, количество комнат и т.д.)
print("\nНазвания признаков:", housing.feature_names)
print("Пример первого образца:", X[0])
print("Тип данных признаков:", X.dtype)

# 3. Подготовка данных
# Разделение на обучающую (70%) и тестовую (30%) выборки со стратификацией
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Масштабирование (StandardScaler) - важно для OMP и рекомендуется для Gamma
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Функция оценки модели для регрессии
def evaluate_regression_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Обучает модель регрессии и возвращает метрики MSE, MAE, R2"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return mse, mae, r2, y_pred

# 5. Обучение на сырых данных
print("ОБУЧЕНИЕ НА СЫРЫХ ДАННЫХ (без масштабирования)")

# GammaRegressor (сырые данные)
gamma_raw = GammaRegressor(max_iter=1000, alpha=0.1)
mse_gamma_raw, mae_gamma_raw, r2_gamma_raw, y_pred_gamma_raw = evaluate_regression_model(
    gamma_raw, X_train, X_test, y_train, y_test, "GammaRegressor (сырые данные)"
)

# OrthogonalMatchingPursuitCV (сырые данные) - ИСПРАВЛЕНО: max_iter = число признаков
omp_raw = OrthogonalMatchingPursuitCV(cv=5, max_iter=8)   # было max_iter=10
mse_omp_raw, mae_omp_raw, r2_omp_raw, y_pred_omp_raw = evaluate_regression_model(
    omp_raw, X_train, X_test, y_train, y_test, "OrthogonalMatchingPursuitCV (сырые данные)"
)

# 6. Обучение на масштабированных данных
print("ОБУЧЕНИЕ НА МАСШТАБИРОВАННЫХ ДАННЫХ (StandardScaler)")

# GammaRegressor (масштабированные)
gamma_scaled = GammaRegressor(max_iter=1000, alpha=0.1)
mse_gamma_sc, mae_gamma_sc, r2_gamma_sc, y_pred_gamma_sc = evaluate_regression_model(
    gamma_scaled, X_train_scaled, X_test_scaled, y_train, y_test,
    "GammaRegressor (масштабированные данные)"
)

# OrthogonalMatchingPursuitCV (масштабированные)
omp_scaled = OrthogonalMatchingPursuitCV(cv=5, max_iter=8)
mse_omp_sc, mae_omp_sc, r2_omp_sc, y_pred_omp_sc = evaluate_regression_model(
    omp_scaled, X_train_scaled, X_test_scaled, y_train, y_test,
    "OrthogonalMatchingPursuitCV (масштабированные данные)"
)

# 7. Кросс-валидация на масштабированных данных
print("КРОСС-ВАЛИДАЦИЯ (3-fold) на масштабированных данных")

kf = KFold(n_splits=3, shuffle=True, random_state=42)

cv_mse_gamma = -cross_val_score(GammaRegressor(max_iter=1000, alpha=0.1),
                                X_train_scaled, y_train, cv=kf,
                                scoring='neg_mean_squared_error')

cv_mse_omp = -cross_val_score(OrthogonalMatchingPursuitCV(cv=5, max_iter=8),
                              X_train_scaled, y_train, cv=kf,
                              scoring='neg_mean_squared_error')


print(f"GammaRegressor CV MSE: {cv_mse_gamma.mean():.4f} (+/- {cv_mse_gamma.std():.4f})")
print(f"OrthogonalMatchingPursuitCV CV MSE: {cv_mse_omp.mean():.4f} (+/- {cv_mse_omp.std():.4f})")

# 8. Визуализация результатов

# 8.1 Сравнение метрик моделей (R2)
models = ['Gamma raw', 'OMP raw', 'Gamma scaled', 'OMP scaled']
r2_scores = [r2_gamma_raw, r2_omp_raw, r2_gamma_sc, r2_omp_sc]
mse_scores = [mse_gamma_raw, mse_omp_raw, mse_gamma_sc, mse_omp_sc]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# R2 score
bars = ax1.bar(models, r2_scores, color=['gray', 'lightblue', 'gray', 'lightblue'])
bars[2].set_color('steelblue')
bars[3].set_color('steelblue')
ax1.set_ylim(0, 1)
ax1.set_ylabel('R² Score')
ax1.set_title('Сравнение R² классификаторов')
for bar, r2 in zip(bars, r2_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{r2:.3f}', ha='center', va='bottom')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# MSE score
bars = ax2.bar(models, mse_scores, color=['gray', 'lightblue', 'gray', 'lightblue'])
bars[2].set_color('steelblue')
bars[3].set_color('steelblue')
ax2.set_ylabel('MSE (ниже = лучше)')
ax2.set_title('Сравнение MSE классификаторов')
for bar, mse in zip(bars, mse_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{mse:.3f}', ha='center', va='bottom')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# 8.2 График предсказаний vs истинных значений 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# GammaRegressor
axes[0].scatter(y_test, y_pred_gamma_sc, alpha=0.5, edgecolors='k', s=20)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Истинные значения')
axes[0].set_ylabel('Предсказанные значения')
axes[0].set_title(f'GammaRegressor (R²={r2_gamma_sc:.3f})')

# OrthogonalMatchingPursuitCV
axes[1].scatter(y_test, y_pred_omp_sc, alpha=0.5, edgecolors='k', s=20)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Истинные значения')
axes[1].set_ylabel('Предсказанные значения')
axes[1].set_title(f'OMPCV (R²={r2_omp_sc:.3f})')

plt.tight_layout()
plt.show()

# 8.3 Анализ остатков
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

residuals_gamma = y_test - y_pred_gamma_sc
residuals_omp = y_test - y_pred_omp_sc

# Гистограмма остатков для Gamma
axes[0].hist(residuals_gamma, bins=30, alpha=0.7, color='steelblue', edgecolor='k')
axes[0].axvline(x=0, color='red', linestyle='--')
axes[0].set_xlabel('Остатки')
axes[0].set_ylabel('Частота')
axes[0].set_title('GammaRegressor: распределение остатков')

# Гистограмма остатков для OMP
axes[1].hist(residuals_omp, bins=30, alpha=0.7, color='lightblue', edgecolor='k')
axes[1].axvline(x=0, color='red', linestyle='--')
axes[1].set_xlabel('Остатки')
axes[1].set_ylabel('Частота')
axes[1].set_title('OMPCV: распределение остатков')

plt.tight_layout()
plt.show()

