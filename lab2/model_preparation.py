from sklearn.model_selection import train_test_split
from model_preprocessing import data_preparation
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle  # для сохранения модели

model = RandomForestClassifier(max_features='log2', n_estimators=300, random_state=73)


df = data_preparation('train.csv')
X, y = df.drop(columns=['Recurred']), df['Recurred']

# разбиваем на тестовую и валидационную
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.3,
                                                  random_state=73)
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl', "wb"))  # сохраняем модель

print('Модель сохранена')
prediction = model.predict(X_val)
print("accuracy:", metrics.accuracy_score(y_val, prediction))