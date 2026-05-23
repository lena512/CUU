
"""
Лабораторная работа №4
Вариант: OneVsRestClassifier, Perceptron
Датасет: load_digits (рукописные цифры)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

#  Конфигурация 
RANDOM_STATE = 42
TEST_SIZE = 0.3
MAX_ITER = 1000
TOL = 1e-3
CV_FOLDS = 3
N_CLUSTERS = 10
TSNE_PERPLEXITY = 30

# 1. Загрузка и подготовка данных 
def load_and_prepare_data():
    """
    Загружает датасет digits, выводит базовую информацию,
    разделяет на обучающую (70%) и тестовую (30%) выборки
    со стратификацией и масштабирует признаки.
    Возвращает:
        X_train, X_test, y_train, y_test - сырые данные
        X_train_scaled, X_test_scaled - масштабированные данные
        scaler - обученный StandardScaler
    """
    digits = load_digits()
    X, y = digits.data, digits.target

    print("Размерность матрицы признаков:", X.shape)
    print("Уникальные классы:", np.unique(y))
    print("Распределение классов:\n", pd.Series(y).value_counts().sort_index())
    print("\nПример признаков первого образца:", X[0])
    print("Тип данных признаков:", X.dtype)

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Масштабирование (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


# 2. Оценка модели 
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Обучает модель и возвращает accuracy, матрицу ошибок и предсказания.
    Также печатает classification report.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} - Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm, y_pred


#3. Обучение и сравнение моделей 
def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_name="сырые данные"):
    """
    Обучает Perceptron и OneVsRestClassifier(Perceptron) на переданных данных.
    Возвращает словарь с результатами.
    """
    # Perceptron
    perceptron = Perceptron(max_iter=MAX_ITER, random_state=RANDOM_STATE, tol=TOL)
    acc_perc, cm_perc, _ = evaluate_model(
        perceptron, X_train, X_test, y_train, y_test,
        f"Perceptron ({dataset_name})"
    )
    
    # OneVsRestClassifier + Perceptron
    ovr = OneVsRestClassifier(Perceptron(max_iter=MAX_ITER, random_state=RANDOM_STATE, tol=TOL))
    acc_ovr, cm_ovr, _ = evaluate_model(
        ovr, X_train, X_test, y_train, y_test,
        f"OneVsRestClassifier + Perceptron ({dataset_name})"
    )
    
    return {"perceptron": (acc_perc, cm_perc), "ovr": (acc_ovr, cm_ovr)}


#  4. Кросс-валидация 
def cross_validate_models(X_train, y_train):
    """
    Выполняет 3-кратную кросс-валидацию для двух моделей
    на масштабированных данных и печатает результаты.
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    perceptron = Perceptron(max_iter=MAX_ITER, random_state=RANDOM_STATE, tol=TOL)
    cv_scores_perc = cross_val_score(perceptron, X_train, y_train, cv=kf, scoring='accuracy')
    
    ovr = OneVsRestClassifier(Perceptron(max_iter=MAX_ITER, random_state=RANDOM_STATE, tol=TOL))
    cv_scores_ovr = cross_val_score(ovr, X_train, y_train, cv=kf, scoring='accuracy')
    
    print(f"\nPerceptron CV accuracy: {cv_scores_perc.mean():.4f} (+/- {cv_scores_perc.std():.4f})")
    print(f"OneVsRestClassifier CV accuracy: {cv_scores_ovr.mean():.4f} (+/- {cv_scores_ovr.std():.4f})")


#  5. Визуализация 
def plot_accuracy_comparison(accuracies, model_names):
    """
    Строит столбчатую диаграмму сравнения точности моделей.
    accuracies: список float, model_names: список str.
    """
    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, accuracies, color=['gray', 'lightblue', 'gray', 'lightblue'])
    bars[2].set_color('steelblue')
    bars[3].set_color('steelblue')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Сравнение точности классификаторов на сырых и масштабированных данных')
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(cm_raw, cm_scaled, title_raw="OvR, сырые данные", title_scaled="OvR, масштабированные данные"):
    """
    Отображает две тепловые карты матриц ошибок.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Матрица ошибок ({title_raw})')
    axes[0].set_xlabel('Предсказанный класс')
    axes[0].set_ylabel('Истинный класс')
    
    sns.heatmap(cm_scaled, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title(f'Матрица ошибок ({title_scaled})')
    axes[1].set_xlabel('Предсказанный класс')
    axes[1].set_ylabel('Истинный класс')
    plt.tight_layout()
    plt.show()


#  6. Кластеризация и визуализация
def perform_clustering(X_train_scaled, y_train):
    """
    Выполняет кластеризацию KMeans, оценивает качество (silhouette, ARI),
    визуализирует центроиды и строит t-SNE проекции.
    """
    # KMeans
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init='auto')
    clusters = kmeans.fit_predict(X_train_scaled)
    
    # Оценка
    sil_score = silhouette_score(X_train_scaled, clusters)
    ari_score = adjusted_rand_score(y_train, clusters)
    print(f"\nSilhouette Score: {sil_score:.4f}")
    print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
    
    # Центроиды кластеров
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    centroids = kmeans.cluster_centers_
    for i, ax in enumerate(axes.flat):
        ax.imshow(centroids[i].reshape(8, 8), cmap='gray')
        ax.set_title(f'Кластер {i}')
        ax.axis('off')
    plt.suptitle('Центроиды кластеров KMeans')
    plt.tight_layout()
    plt.show()
    
    # t-SNE проекция
    print("Вычисление t-SNE проекции (может занять несколько секунд)...")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=TSNE_PERPLEXITY)
    X_tsne = tsne.fit_transform(X_train_scaled)
    
    # Цвет по кластерам
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Номер кластера')
    plt.title('t-SNE визуализация результатов кластеризации KMeans (10 кластеров)')
    plt.xlabel('t-SNE компонента 1')
    plt.ylabel('t-SNE компонента 2')
    plt.show()
    
    # Цвет по истинным меткам
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Истинная цифра')
    plt.title('t-SNE визуализация: истинные метки классов')
    plt.xlabel('t-SNE компонента 1')
    plt.ylabel('t-SNE компонента 2')
    plt.show()


#Основной блок 
def main():
    # Загрузка и подготовка данных
    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled, _) = load_and_prepare_data()
    
    # Обучение на сырых данных
    print("ОБУЧЕНИЕ НА СЫРЫХ ДАННЫХ (без масштабирования)")
    raw_results = train_and_evaluate_models(X_train, X_test, y_train, y_test, "сырые данные")
    
    # Обучение на масштабированных данных
    print("ОБУЧЕНИЕ НА МАСШТАБИРОВАННЫХ ДАННЫХ (StandardScaler)")
    scaled_results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, "масштабированные данные")
    
    # Сбор метрик для визуализации
    accuracies = [
        raw_results["perceptron"][0],   # Perceptron raw
        raw_results["ovr"][0],          # OvR raw
        scaled_results["perceptron"][0], # Perceptron scaled
        scaled_results["ovr"][0]         # OvR scaled
    ]
    model_names = ['Perceptron raw', 'OvR raw', 'Perceptron scaled', 'OvR scaled']
    plot_accuracy_comparison(accuracies, model_names)
    
    # Матрицы ошибок (для OvR моделей)
    plot_confusion_matrices(
        raw_results["ovr"][1],
        scaled_results["ovr"][1],
        "OvR, сырые данные",
        "OvR, масштабированные данные"
    )
    
    # Кросс-валидация на масштабированных данных
    print("КРОСС-ВАЛИДАЦИЯ (3-fold) на масштабированных данных")
    cross_validate_models(X_train_scaled, y_train)
    
    # Кластеризация и визуализация
    perform_clustering(X_train_scaled, y_train)


if __name__ == "__main__":
    main()