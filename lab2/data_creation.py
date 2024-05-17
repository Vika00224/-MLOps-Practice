import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/Thyroid_Diff.csv')
# разбиваем на тренировочную и валидационную
x_train, x_val, = train_test_split(df, test_size=0.2,random_state=42)
# сохранение данных в csv
x_train.to_csv('train.csv', index=False)
x_val.to_csv('test.csv', index=False)

print('Tренировочные данные: сохранены')
print('Тестовые данные: сохранены')