import numpy as np
import matplotlib.pyplot as plt

# Функция для определения выбросов на основе MAD
def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh

# Создание набора данных с выбросами
np.random.seed(42)
data = np.random.normal(0, 1, 100)
data = np.append(data, [5, 10, -3, -4, 7])  # Добавляем выбросы

# Применение алгоритма MAD
outliers_mad = mad_based_outlier(data)
outliers_indices = np.where(outliers_mad)[0]

# Удаление выбросов по порогу
threshold = 3
z_scores = (data - np.mean(data)) / np.std(data)
data_cleaned_threshold = data[abs(z_scores) < threshold]

# Замена выбросов на медианное значение
median_value = np.median(data)
data_cleaned_median = np.where(abs(z_scores) > threshold, median_value, data)

# Визуализация результатов
plt.figure(figsize=(18, 6))

# Исходные данные и выбросы
plt.subplot(1, 3, 1)
plt.plot(data, 'o', label='Исходные данные')
plt.plot(outliers_indices, data[outliers_indices], 'rx', label='Выбросы MAD', markersize=10)
plt.title('Исходные данные с выбросами')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()

# Данные после удаления по порогу
plt.subplot(1, 3, 2)
plt.plot(data_cleaned_threshold, 'o', label='После удаления по порогу')
plt.title('Данные после удаления по порогу')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()

# Данные после замены на медиану
plt.subplot(1, 3, 3)
plt.plot(data_cleaned_median, 'o', label='После замены на медиану')
plt.title('Данные после замены на медиану')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()

plt.tight_layout()
plt.show()
