
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X_train = pd.read_csv("../data/gold/X_train.csv")
X_test = pd.read_csv("../data/gold/X_test.csv")
y_train = pd.read_csv("../data/gold/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/gold/y_test.csv").values.ravel()

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Previsões feitas pelo modelo:", y_pred)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("Matriz de Confusão:", confusion_matrix(y_test, y_pred))
print("Relatório de Classificação:", classification_report(y_test, y_pred))

os.makedirs("../models", exist_ok=True)
joblib.dump(modelo, "../models/logistic_model.pkl")
print("Modelo salvoo")

pd.DataFrame(y_pred, columns=["Previsões"]).to_csv("../data/gold/y_pred.csv", index=False)
print("Previsões salva")

