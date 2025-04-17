import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def main():

    X_train = pd.read_csv("../data/gold/X_train.csv")
    X_test = pd.read_csv("../data/gold/X_test.csv")
    y_train = pd.read_csv("../data/gold/y_train.csv").values.ravel()
    y_test = pd.read_csv("../data/gold/y_test.csv").values.ravel()

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification = classification_report(y_test, y_pred)

    print("=" * 50)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("=" * 50)

    print("Confusion Matrix:")
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Good Credit', 'Bad Credit'],
                yticklabels=['Good Credit', 'Bad Credit'])
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Credit Risk Assessment')
    plt.show()

    print("Classification Report:")
    print("=" * 50)
    print(classification)
    print("=" * 50)
    print("Metrics Description:")
    print("  Precision: Of all instances predicted to be in a class, how many were correct.")
    print("  Recall: Of all the real instances of a class, how many were correctly classified.")
    print("  F1-score: The harmonic mean between precision and recall.")
    print("  Support: Number of real instances of a given class.")

    print("\nData Description:")
    print("The model was tested with the test subset of the HMEQ dataset, "
          "that contains data regarding home equity loans.")

    results_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    os.makedirs("../data/gold/", exist_ok=True)
    results_df.to_csv("../data/gold/predictions_vs_real.csv", index=False)
    print("\nPredictions vs Real values saved to: ../data/gold/predictions_vs_real.csv")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_proba, alpha=0.5)
    plt.xlabel("Actual Labels (0: Good Credit, 1: Bad Credit)")
    plt.ylabel("Predicted Probabilities (Probability of Bad Credit)")
    plt.title("Scatter Plot of Predicted Probabilities vs. Actual Labels")
    plt.xticks([0, 1], ['Good Credit', 'Bad Credit'])
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)
    plt.show()

    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/logistic_model.pkl")
    print("\nModel saved to: ../models/logistic_model.pkl")




    feature_index = 0

    X_plot = X_test.iloc[:, feature_index].values.reshape(-1, 1)

    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X_plot, y_test)
    proba = pipe.predict_proba(X_plot)[:, 1]

    plt.figure(figsize=(10, 5))
    plt.scatter(X_plot, y_test, color='red', label='Real')
    plt.scatter(X_plot, proba, color='blue', alpha=0.5, label='Probabilidade Prevista')
    plt.xlabel(X_test.columns[feature_index])
    plt.ylabel('Probabilidade de Crédito Ruim')
    plt.title('Curva de Regressão Logística (1 Feature)')
    plt.legend()
    plt.grid(True)
    plt.show()


    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.title('Curva Precision-Recall')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    import numpy as np

    main()
