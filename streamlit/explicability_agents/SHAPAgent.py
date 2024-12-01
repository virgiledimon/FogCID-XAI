import shap
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SHAPAgent:
    def __init__(self, model, X_test):
        self.model = model
        self.scaler = StandardScaler()
        self.X_test = X_test

    # Méthode pour expliquer une instance spécifique avec SHAP
    def explain_instance(self):

        if not isinstance(self.X_test, pd.DataFrame):
          feature_names = ['lambda_threshold', 'L_value']
          self.X_test = pd.DataFrame(self.X_test, columns=feature_names)

        # Sélection de la première instance de test
        X_instance = self.X_test.iloc[0]
        print(X_instance)
        # Créer l'explainer SHAP basé sur le modèle (assurez-vous que c'est un modèle d'arbre)
        explainer = shap.TreeExplainer(self.model)
        # Calcul des valeurs SHAP pour l'instance
        shap_values = explainer.shap_values(X_instance)
        print(shap_values)
        print(explainer.expected_value)
        print("\n")
        print(f"Base value: {explainer.expected_value[1]}")
        # Initialisation de JavaScript pour les graphiques interactifs
        return explainer.expected_value[1], shap_values[..., 1], self.X_test.columns
    
    # Méthode pour expliquer le modèle entier avec SHAP
    def explain_model(self):

        if not isinstance(self.X_test, pd.DataFrame):
          feature_names = ['lambda_threshold', 'L_value']
          self.X_test = pd.DataFrame(self.X_test, columns=feature_names)

        # Créer l'explainer SHAP basé sur le modèle
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test, approximate=True)

        # Affichage des valeurs SHAP sous forme de tableau
        self.display_shap_values_table(shap_values, self.X_test)

        # Afficher le summary plot et le bar plot
        accept_shap_values = shap_values[..., 1]
        
        return accept_shap_values, feature_names, self.X_test
        
        
    # Méthode pour afficher les valeurs SHAP dans un tableau
    def display_shap_values_table(self, shap_values, X_test):
        # Sélectionner les valeurs SHAP pour la classe d'acceptation
        accept_shap_values = shap_values[...,1]  # Utilisation de la classe 'Accept'
        # Créer un DataFrame pour afficher les valeurs SHAP
        shap_df = pd.DataFrame(accept_shap_values, columns=X_test.columns)
        print("\nTableau des valeurs SHAP pour les décisions 'Accept':")
        print(shap_df.head())  # Afficher les premières lignes du tableau
