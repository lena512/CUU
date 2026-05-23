import math
import random
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Агломеративная кластеризация 
def hierarchical_clustering(lat, lon, k):
    n = len(lat)
    clusters = [[i] for i in range(n)]
    # матрица расстояний
    dist_matrix = [[math.hypot(lat[i]-lat[j], lon[i]-lon[j]) for j in range(n)] for i in range(n)]
    
    while len(clusters) > k:
        # поиск пары кластеров с минимальным single‑link расстоянием
        min_dist = float('inf')
        merge_i, merge_j = -1, -1
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                d = min(dist_matrix[a][b] for a in clusters[i] for b in clusters[j])
                if d < min_dist:
                    min_dist = d
                    merge_i, merge_j = i, j
        clusters[merge_i].extend(clusters[merge_j])
        del clusters[merge_j]
    
    # Вычисляем центроиды (как средние координаты) для единообразия
    points = np.array(list(zip(lat, lon)))
    centroids = []
    for cl in clusters:
        if cl:
            cen = np.mean(points[cl], axis=0)
            centroids.append(cen)
        else:
            centroids.append([np.nan, np.nan])
    return clusters, np.array(centroids)

#2. K‑means (ручная реализация)
def kmeans_manual(lat, lon, k, max_iter=100):
    n = len(lat)
    points = np.array(list(zip(lat, lon)))
    min_lat, max_lat = min(lat), max(lat)
    min_lon, max_lon = min(lon), max(lon)
    # случайная инициализация центроидов
    centroids = np.array([[random.uniform(min_lat, max_lat),
                           random.uniform(min_lon, max_lon)] for _ in range(k)])
    
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for i, p in enumerate(points):
            dists = [math.hypot(p[0]-c[0], p[1]-c[1]) for c in centroids]
            cl_idx = np.argmin(dists)
            clusters[cl_idx].append(i)
        new_centroids = []
        for c in range(k):
            if clusters[c]:
                avg_lat = np.mean([points[i][0] for i in clusters[c]])
                avg_lon = np.mean([points[i][1] for i in clusters[c]])
                new_centroids.append([avg_lat, avg_lon])
            else:
                # пустой кластер – случайная точка
                new_centroids.append([random.uniform(min_lat, max_lat),
                                      random.uniform(min_lon, max_lon)])
        new_centroids = np.array(new_centroids)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

#3. K‑means из scikit‑learn
def kmeans_sklearn(lat, lon, k):
    X = np.array(list(zip(lat, lon)))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    clusters = [[] for _ in range(k)]
    for i, lbl in enumerate(labels):
        clusters[lbl].append(i)
    return clusters, km.cluster_centers_

# Вспомогательные функции оценки
def inertia(clusters, lat, lon):
    """Сумма квадратов расстояний от точек до центров их кластеров."""
    points = np.array(list(zip(lat, lon)))
    total = 0.0
    for cl in clusters:
        if not cl:
            continue
        center = np.mean(points[cl], axis=0)
        for idx in cl:
            total += np.sum((points[idx] - center)**2)
    return total

def silhouette_score_own(labels, lat, lon):
    """Индекс силуэта (требует метки для каждой точки)."""
    X = np.array(list(zip(lat, lon)))
    if len(set(labels)) < 2:
        return -1.0
    return silhouette_score(X, labels)

def time_algorithm(alg_func, lat, lon, k):
    """Замеряет время выполнения алгоритма и возвращает (clusters, время)."""
    start = time.perf_counter()
    clusters, _ = alg_func(lat, lon, k)   # все функции возвращают (clusters, центроиды)
    elapsed = time.perf_counter() - start
    return clusters, elapsed

# Генерация тестовых конфигураций
def generate_test_configs():
    configs = []
    # 1. Три компактные группы
    lat1 = [10,10,10, 30,30,30, 50,50,50]
    lon1 = [10,11,12, 30,31,32, 50,51,52]
    configs.append(('3 компактные группы', lat1, lon1, 3))
    # 2. Случайные точки
    random.seed(42)
    lat2 = [random.uniform(0,100) for _ in range(15)]
    lon2 = [random.uniform(0,100) for _ in range(15)]
    configs.append(('Случайные точки', lat2, lon2, 3))
    # 3. Линейная структура
    lat3 = list(range(20))
    lon3 = [2*x for x in range(20)]
    configs.append(('Линейная структура', lat3, lon3, 2))
    # 4. Один выброс
    lat4 = [10,11,12,13,14, 100]
    lon4 = [10,11,12,13,14, 100]
    configs.append(('Один выброс', lat4, lon4, 2))
    # 5. Исходные города
    lat5 = [52.5, 32.1, 12.5, 87.5, 20.1]
    lon5 = [34.2, 12.1, 90.2, 80.1, 89.3]
    configs.append(('Исходные города', lat5, lon5, 2))
    return configs

#Запуск эксперимента 
def run_experiment():
    configs = generate_test_configs()
    results = []
    for name, lat, lon, k in configs:
        print(f"\n=== {name} (K={k}) ===")
        
        # Агломеративный
        cl_h, t_h = time_algorithm(hierarchical_clustering, lat, lon, k)
        inert_h = inertia(cl_h, lat, lon)
        labels_h = [0]*len(lat)
        for ci, cl in enumerate(cl_h):
            for idx in cl:
                labels_h[idx] = ci
        sil_h = silhouette_score_own(labels_h, lat, lon)
        
        # K‑means ручной
        cl_m, t_m = time_algorithm(kmeans_manual, lat, lon, k)
        inert_m = inertia(cl_m, lat, lon)
        labels_m = [0]*len(lat)
        for ci, cl in enumerate(cl_m):
            for idx in cl:
                labels_m[idx] = ci
        sil_m = silhouette_score_own(labels_m, lat, lon)
        
        # K‑means sklearn
        cl_s, t_s = time_algorithm(kmeans_sklearn, lat, lon, k)
        inert_s = inertia(cl_s, lat, lon)
        labels_s = [0]*len(lat)
        for ci, cl in enumerate(cl_s):
            for idx in cl:
                labels_s[idx] = ci
        sil_s = silhouette_score_own(labels_s, lat, lon)
        
        results.append({
            'config': name, 'k': k,
            'hier_inertia': inert_h, 'hier_silhouette': sil_h, 'hier_time': t_h,
            'manual_inertia': inert_m, 'manual_silhouette': sil_m, 'manual_time': t_m,
            'sklearn_inertia': inert_s, 'sklearn_silhouette': sil_s, 'sklearn_time': t_s
        })
        
        print(f"Агломеративный: inertia={inert_h:.2f}, silhouette={sil_h:.3f}, time={t_h:.5f}s")
        print(f"K-means ручной: inertia={inert_m:.2f}, silhouette={sil_m:.3f}, time={t_m:.5f}s")
        print(f"K-means sklearn: inertia={inert_s:.2f}, silhouette={sil_s:.3f}, time={t_s:.5f}s")
    
    return results

if __name__ == "__main__":
    run_experiment()