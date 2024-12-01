from sklearn.inspection import permutation_importance
import numpy as np

class PFIAgent:
    def __init__(self, model, X_test_data, y_test_data):
        self.model = model
        self.X_test = X_test_data  # Initialisation des données normalisées
        self.y_test = y_test_data  # Initialisation des étiquettes

    def compute_importance(self):
        if self.X_test is None or self.y_test is None:
            print("Les données ne sont pas encore disponibles. Veuillez appeler 'get_normalized_decision_data' avant.")
            return

        result = permutation_importance(self.model, self.X_test, self.y_test, n_repeats=10, random_state=42)
        return result.importances_mean

    def plot_importance(self, importances):
        # Créer un graphique des importances
        feature_names = ['lambda_threshold', 'L_value']  # Noms des caractéristiques
        indices = np.argsort(importances)[::-1]
        return feature_names, indices