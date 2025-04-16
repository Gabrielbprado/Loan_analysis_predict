import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

X_train = pd.read_csv("../data/gold/X_train.csv")
X_test = pd.read_csv("../data/gold/X_test.csv")
y_train = pd.read_csv("../data/gold/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/gold/y_test.csv").values.ravel()

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
matriz_confusao = confusion_matrix(y_test, y_pred)
relatorio_classificacao = classification_report(y_test, y_pred)


print("" + "="*50)
print(f"Acurácia do Modelo: {acuracia*100:.2f}%")
print("="*50)

print("Matriz de Confusão:")
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Predição')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

print("Relatório de Classificação:")
print("="*50)
print(relatorio_classificacao)
print("="*50)

os.makedirs("../models", exist_ok=True)
joblib.dump(modelo, "../models/logistic_model.pkl")
print("Modelo salvo")
