import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

# открываем a CSV файл
df = pd.read_csv('data/datasets/wine.csv')

# Удаляем целевую переменную
X = df.drop('target', axis=1)
y = df['target']

# Выбираем k важных характеристик
k = 5
selector = SelectKBest(chi2, k=k)
X_new = selector.fit_transform(X, y)

# Получаем названия наиболее важных характеристик
mask = selector.get_support()
new_features = X.columns[mask]

print("Важные характеристики:", list(new_features))

# Делим данные на тестовую и валидационную выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, test_size=0.2, random_state=42, shuffle=True)

# Нормализация и стандартизация данных
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создание каталогов для хранения выборок
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/test', exist_ok=True)

# Сохраняем CSV файлы
pd.DataFrame(X_train_scaled, columns=new_features).to_csv(
    'data/train/X_train.csv', index=False)
pd.DataFrame(y_train, columns=['target']).to_csv(
    'data/train/y_train.csv', index=False)
pd.DataFrame(X_test_scaled, columns=new_features).to_csv(
    'data/test/X_test.csv', index=False)
pd.DataFrame(y_test, columns=['target']).to_csv(
    'data/test/y_test.csv', index=False)