from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

class InterpretabilityAgent:
    
    def __init__(self):
        self.scaler_minmax = MinMaxScaler()
        self.scaler_standard = StandardScaler()
        self.model_decision = DecisionTreeClassifier()  # Modèle pour les décisions (arbre de décision)

    # Méthode pour normaliser les données de décisions
    def normalize_decision_data(self, X_train, X_test):
        X_train_normalized = self.scaler_standard.fit_transform(X_train)
        X_test_normalized = self.scaler_standard.transform(X_test)
        return X_train_normalized, X_test_normalized

    # Méthode pour entraîner le modèle de décisions
    def train_decision_model(self, X_train, y_train):
        self.model_decision.fit(X_train, y_train)
        print("Modèle de décisions entraîné avec succès.")

    # Méthode pour faire des prédictions sur les décisions
    def predict_decision(self, X_test):
        return self.model_decision.predict(X_test)

    # Méthode pour évaluer les prédictions
    def evaluate_model(self, X_test, y_test):
        y_pred = self.predict_decision(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Précision du modèle : {accuracy:.2f}")
        return accuracy

    # Méthode pour afficher l'arbre de décision
    def plot_decision_tree(self):
        if hasattr(self.model_decision, 'tree_'):
            plt.figure(figsize=(20, 10))
            plot_tree(self.model_decision, filled=True,
                      feature_names=['lambda_threshold', 'l_value'],
                      class_names=['Reject', 'Accept'])
            plt.show()
        else:
            print("Le modèle n'est pas encore entraîné. Entraînez d'abord le modèle avant de l'afficher.")

    