import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
import numpy as np

class LIMEAgent:
    def __init__(self, model, normalized_test_data):
        self.model = model
        self.X_lime_test = normalized_test_data  # Initialisation des données normalisées

    # Méthode pour expliquer une instance spécifique avec LIME
    def explain_instance(self, X_instance):
        if self.X_lime_test is None:
            print("Les données ne sont pas encore disponibles. Veuillez appeler 'get_normalized_decision_data' avant.")
            return

        plt.style.use('default')  # Appliquer un style avec fond blanc

        # Initialisation de l'explainer LIME
        explainer = LimeTabularExplainer(
            self.X_lime_test,
            feature_names=['lambda_threshold', 'L_value'],  # Les noms des caractéristiques
            class_names=['Reject', 'Accept'],
            mode='classification',
            verbose=True,
        )

        print("\n")

        # Expliquer l'instance donnée (qui doit être normalisée)
        exp = explainer.explain_instance(X_instance, self.model.predict_proba, num_features=2)

        return exp

        # fig.savefig("lime_explanation_white_bg.png", bbox_inches='tight', facecolor='white')

        print("\n")