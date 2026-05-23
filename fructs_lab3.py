import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

#  Подготовка исходного набора данных 
#  Классы: Фрукты (сладкие, любой хруст),
#          Овощи (несладкие, хрустящие),
#          Протеин (несладкие, нехрустящие)

np.random.seed(42)
n_samples = 50

# Генерируем данные для Овощей и Протеина 
# Овощи: низкая сладость, высокая хруст
veg_sweet = np.random.normal(1.5, 1.2, n_samples)
veg_crunch = np.random.normal(8.0, 1.2, n_samples)
# Протеин: низкая сладость, низкий хруст
prot_sweet = np.random.normal(1.5, 1.2, n_samples)
prot_crunch = np.random.normal(2.0, 1.2, n_samples)

# Фрукты: сладкие, хруст может быть любым (от низкого до высокого)
fruit_sweet = np.random.normal(8.0, 1.0, n_samples)   
fruit_crunch = np.random.uniform(1.0, 9.0, n_samples) + np.random.normal(0, 0.3, n_samples)  # равномерно + шум

# Собираем в DataFrame
data_3class = []
# добавляем Овощи
for s, c in zip(veg_sweet, veg_crunch):
    data_3class.append([s, c, 'Овощ'])
# Протеин
for s, c in zip(prot_sweet, prot_crunch):
    data_3class.append([s, c, 'Протеин'])
# Фрукты
for s, c in zip(fruit_sweet, fruit_crunch):
    data_3class.append([s, c, 'Фрукт'])

df_3 = pd.DataFrame(data_3class, columns=['сладость', 'хруст', 'класс'])
print("Сгенерированный набор данных (3 класса):")
print(df_3.head(10).to_string(index=False), "\n...")
print(f"Всего записей: {len(df_3)}")

X_3 = df_3[['сладость', 'хруст']].values
y_3 = df_3['класс'].values

# Тестовые примеры для трёх классов
manual_test_3 = {
    'продукт': ['Персик', 'Тофу', 'Сельдерей'],
    'сладость': [8, 1, 1],
    'хруст':    [3, 1, 9],
    'реальный_класс': ['Фрукт', 'Протеин', 'Овощ']
}
df_test3 = pd.DataFrame(manual_test_3)
X_test3 = df_test3[['сладость', 'хруст']].values
y_test3_real = df_test3['реальный_класс'].values

print("\nТестовые примеры (из задания):")
print(df_test3[['продукт', 'сладость', 'хруст']].to_string(index=False))

#   Собственная реализация k-NN

class KNN_from_scratch:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)

# Проверка собственного классификатора (k=3)
k_init = 3
my_knn = KNN_from_scratch(k=k_init)
my_knn.fit(X_3, y_3)
y_pred_my3 = my_knn.predict(X_test3)

print(f"\n--- Результаты моего k-NN (k={k_init}) ---")
for i, prod in enumerate(df_test3['продукт']):
    print(f"{prod}: предсказан {y_pred_my3[i]}, истинный {y_test3_real[i]}")

# Сравнение с sklearn
sk_knn = KNeighborsClassifier(n_neighbors=k_init)
sk_knn.fit(X_3, y_3)
y_pred_sk3 = sk_knn.predict(X_test3)
print(f"\n--- Результаты sklearn k-NN (k={k_init}) ---")
for i, prod in enumerate(df_test3['продукт']):
    print(f"{prod}: предсказан {y_pred_sk3[i]}, истинный {y_test3_real[i]}")

print(f"\nСовпадение результатов: {np.array_equal(y_pred_my3, y_pred_sk3)}")


#  Кросс-валидация для выбора оптимального k

k_values = range(1, 15)
cv_scores_3 = []
for k in k_values:
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_cv, X_3, y_3, cv=5, scoring='accuracy')
    cv_scores_3.append(scores.mean())
    print(f"k={k:2d}, средняя точность CV: {scores.mean():.4f}")

optimal_k_3 = k_values[np.argmax(cv_scores_3)]
print(f"\nОптимальное k для 3 классов: {optimal_k_3}")

# График кросс-валидации
plt.figure(figsize=(10, 4))
plt.plot(k_values, cv_scores_3, marker='o', linestyle='dashed', color='b')
plt.axvline(x=optimal_k_3, color='r', linestyle='-',
            label=f'Оптимальное k = {optimal_k_3}')
plt.xlabel('Количество соседей (k)')
plt.ylabel('Средняя точность кросс-валидации')
plt.title('Кросс-валидация для выбора k (3 класса, фрукты – сладкие, любой хруст)')
plt.legend()
plt.grid(True)
plt.show()


# Визуализация данных и границ решений (3 класса)

def plot_2d_boundaries(X, y, classifier, title, test_X=None, test_labels=None):
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    unique_cls = np.unique(y)
    cls_to_num = {c: i for i, c in enumerate(unique_cls)}
    Z_num = np.vectorize(cls_to_num.get)(Z).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z_num, alpha=0.3, cmap=plt.cm.Set3)
    colors = ['red', 'green', 'blue']
    for i, cls in enumerate(unique_cls):
        plt.scatter(X[y == cls, 0], X[y == cls, 1],
                    label=cls, color=colors[i], edgecolor='k', s=30)
    if test_X is not None:
        plt.scatter(test_X[:, 0], test_X[:, 1],
                    color='black', marker='X', s=100, label='Тестовые')
        if test_labels is not None:
            for i, txt in enumerate(test_labels):
                plt.annotate(txt, (test_X[i, 0], test_X[i, 1]),
                             textcoords="offset points", xytext=(5,5),
                             ha='center', fontsize=9)
    plt.xlabel('Сладость')
    plt.ylabel('Хруст')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Обучение лучшей модели и визуализация
best_knn3 = KNeighborsClassifier(n_neighbors=optimal_k_3)
best_knn3.fit(X_3, y_3)
plot_2d_boundaries(X_3, y_3, best_knn3,
                   f'k-NN классификация (3 класса, k=3)\nФрукты: сладкие, любой хруст',
                   test_X=X_test3, test_labels=df_test3['продукт'].tolist())

print("\nИтоговый отчёт по тестовым примерам (лучшая модель для 3 классов):")
print(classification_report(y_test3_real, best_knn3.predict(X_test3), zero_division=0))


#  Введение четвёртого класса и нового параметра
#  Добавляем класс "Зерновые" и третий признак "калорийность"

print("РАСШИРЕННЫЙ ЭКСПЕРИМЕНТ: 4 класса, 3 признака")

np.random.seed(123)  # для воспроизводимости
n_samples_4 = 50

# Овощи (низкая сладость, высокий хруст)
veg_sw = np.random.normal(1.5, 1.0, n_samples_4)
veg_cr = np.random.normal(8.0, 1.0, n_samples_4)
veg_cal = np.random.normal(20, 5, n_samples_4)

# Протеин (низкая сладость, низкий хруст)
prot_sw = np.random.normal(1.5, 1.0, n_samples_4)
prot_cr = np.random.normal(2.0, 1.0, n_samples_4)
prot_cal = np.random.normal(120, 10, n_samples_4)

# Фрукты (сладкие, любой хруст)
fruit_sw = np.random.normal(8.0, 1.0, n_samples_4)
fruit_cr = np.random.uniform(1.0, 9.0, n_samples_4) + np.random.normal(0, 0.3, n_samples_4)
fruit_cal = np.random.normal(40, 8, n_samples_4)

# Зерновые (средняя сладость, высокий хруст, высокая калорийность)
grain_sw = np.random.normal(5.5, 1.0, n_samples_4)
grain_cr = np.random.normal(7.5, 1.0, n_samples_4)
grain_cal = np.random.normal(90, 10, n_samples_4)

data_4class = []
for s, c, cal in zip(veg_sw, veg_cr, veg_cal):
    data_4class.append([s, c, cal, 'Овощ'])
for s, c, cal in zip(prot_sw, prot_cr, prot_cal):
    data_4class.append([s, c, cal, 'Протеин'])
for s, c, cal in zip(fruit_sw, fruit_cr, fruit_cal):
    data_4class.append([s, c, cal, 'Фрукт'])
for s, c, cal in zip(grain_sw, grain_cr, grain_cal):
    data_4class.append([s, c, cal, 'Зерновые'])

df_4 = pd.DataFrame(data_4class, columns=['сладость', 'хруст', 'калорийность', 'класс'])
print("\nРасширенный набор данных (4 класса, 3 признака):")
print(df_4.head(10).to_string(index=False), "\n...")
print(f"Всего записей: {len(df_4)}")

X_4 = df_4[['сладость', 'хруст', 'калорийность']].values
y_4 = df_4['класс'].values

# Тестовые примеры для 4 классов
manual_test_4 = {
    'продукт': ['Персик', 'Тофу', 'Сельдерей', 'Гранола'],
    'сладость': [8, 1, 1, 6],
    'хруст':    [3, 1, 9, 8],
    'калорийность': [38, 110, 18, 95],
    'реальный_класс': ['Фрукт', 'Протеин', 'Овощ', 'Зерновые']
}
df_test4 = pd.DataFrame(manual_test_4)
X_test4 = df_test4[['сладость', 'хруст', 'калорийность']].values
y_test4_real = df_test4['реальный_класс'].values

print("\nТестовые примеры (4 класса, 3 признака):")
print(df_test4[['продукт', 'сладость', 'хруст', 'калорийность']].to_string(index=False))

# Собственная реализация k-NN (переиспользуем класс)
my_knn4 = KNN_from_scratch(k=k_init)
my_knn4.fit(X_4, y_4)
y_pred_my4 = my_knn4.predict(X_test4)

print(f"\n--- Мой k-NN (3 признака, k=3) ---")
for i, prod in enumerate(df_test4['продукт']):
    print(f"{prod}: предсказан {y_pred_my4[i]}, истинный {y_test4_real[i]}")

# sklearn k-NN
sk_knn4 = KNeighborsClassifier(n_neighbors=k_init)
sk_knn4.fit(X_4, y_4)
y_pred_sk4 = sk_knn4.predict(X_test4)
print(f"\n--- sklearn k-NN (3 признака, k={k_init}) ---")
for i, prod in enumerate(df_test4['продукт']):
    print(f"{prod}: предсказан {y_pred_sk4[i]}, истинный {y_test4_real[i]}")

print(f"\nСовпадение результатов: {np.array_equal(y_pred_my4, y_pred_sk4)}")

# Кросс-валидация для расширенного набора
cv_scores_4 = []
for k in k_values:
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_cv, X_4, y_4, cv=5, scoring='accuracy')
    cv_scores_4.append(scores.mean())

optimal_k_4 = k_values[np.argmax(cv_scores_4)]
print(f"\nОптимальное k для 4 классов (3 признака): {optimal_k_4}")

plt.figure(figsize=(10, 4))
plt.plot(k_values, cv_scores_4, marker='s', linestyle='dashed', color='m')
plt.axvline(x=optimal_k_4, color='r', linestyle='-',
            label=f'Оптимальное k = {optimal_k_4}')
plt.xlabel('Количество соседей (k)')
plt.ylabel('Средняя точность кросс-валидации')
plt.title('Кросс-валидация (4 класса, 3 признака)')
plt.legend()
plt.grid(True)
plt.show()

# Визуализация расширенных данных
fig = plt.figure(figsize=(14, 5))

# 2D проекция (сладость vs хруст)
ax1 = fig.add_subplot(1, 2, 1)
colors = ['red', 'green', 'blue', 'orange']
markers = ['o', 's', '^', 'D']
for i, cls in enumerate(np.unique(y_4)):
    idx = y_4 == cls
    ax1.scatter(X_4[idx, 0], X_4[idx, 1],
                color=colors[i], marker=markers[i], label=cls, edgecolor='k', alpha=0.6)
ax1.scatter(X_test4[:, 0], X_test4[:, 1], color='black', marker='X', s=100, label='Тест')
for i, txt in enumerate(df_test4['продукт']):
    ax1.annotate(txt, (X_test4[i, 0], X_test4[i, 1]),
                 textcoords="offset points", xytext=(5,5), ha='center')
ax1.set_xlabel('Сладость'); ax1.set_ylabel('Хруст')
ax1.set_title('Проекция на (сладость, хруст)\nФрукты – сладкие, любой хруст')
ax1.legend(); ax1.grid(True)

# 3D-визуализация
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for i, cls in enumerate(np.unique(y_4)):
    idx = y_4 == cls
    ax2.scatter(X_4[idx, 0], X_4[idx, 1], X_4[idx, 2],
                color=colors[i], marker=markers[i], label=cls, alpha=0.6)
ax2.scatter(X_test4[:, 0], X_test4[:, 1], X_test4[:, 2],
            color='black', marker='X', s=100, label='Тест')
ax2.set_xlabel('Сладость'); ax2.set_ylabel('Хруст'); ax2.set_zlabel('Калорийность')
ax2.set_title('Трёхмерное пространство признаков')
ax2.legend()
plt.tight_layout()
plt.show()

# Итоговый отчёт для расширенного эксперимента
best_knn4 = KNeighborsClassifier(n_neighbors=optimal_k_4)
best_knn4.fit(X_4, y_4)
y_pred_best4 = best_knn4.predict(X_test4)
print("\nИтоговый отчёт по тестовым примерам (4 класса, 3 признака):")
print(classification_report(y_test4_real, y_pred_best4, zero_division=0))

