import pandas as pd
import os
from sklearn.datasets import load_wine

# загружаем датасет
wine = load_wine()
X = wine.data  # type: ignore
y = wine.target  # type: ignore

# трансформируем в датафрейм
df = pd.DataFrame(data=X, columns=wine.feature_names)  # type: ignore
df['target'] = y

print(df.info())
print(df.describe())

# создание каталогов для хранения данных
os.makedirs('datasets', exist_ok=True)

# сохранение CSV файла
df.to_csv('datasets/wine.csv', index=False)

print('Данные сохранены')
