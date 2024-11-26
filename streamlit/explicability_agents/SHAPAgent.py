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
        force_plot = shap.force_plot(explainer.expected_value[1], shap_values[..., 1], feature_names=self.X_test.columns, link="logit", matplotlib=False, show=True, figsize=(9,4))
        return f"<head>{shap.getjs()}</head><body style='background-color:white;'>{force_plot.html()}</body>"
    
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
        
        return {
           shap.summary_plot(accept_shap_values, self.X_test),
           shap.summary_plot(accept_shap_values, self.X_test, plot_type="bar"),
           shap.dependence_plot("lambda_threshold", accept_shap_values, self.X_test),
           shap.dependence_plot("L_value", accept_shap_values, self.X_test)
        }
        

    # Méthode pour afficher les valeurs SHAP dans un tableau
    def display_shap_values_table(self, shap_values, X_test):
        # Sélectionner les valeurs SHAP pour la classe d'acceptation
        accept_shap_values = shap_values[...,1]  # Utilisation de la classe 'Accept'
        # Créer un DataFrame pour afficher les valeurs SHAP
        shap_df = pd.DataFrame(accept_shap_values, columns=X_test.columns)
        print("\nTableau des valeurs SHAP pour les décisions 'Accept':")
        print(shap_df.head())  # Afficher les premières lignes du tableau


# Exemple d'utilisation
# if __name__ == "__main__":
#     # Supposons que db_agent est déjà initialisé et connecté à la base de données
#     db_agent = PostgreSQLAgent(db_name='', user='postgres', password='postgres')

#     # Créer l'agent SHAP en utilisant le modèle déjà entraîné
#     shap_agent = SHAPAgent(model=interpret_agent.model_decision, db_agent=db_agent)

#     # Récupérer et normaliser les données de décisions depuis la base
#     shap_agent.get_normalized_decision_data()

#     # Expliquer le modèle entier
#     shap_agent.explain_model()

#     # Instance
#     shap_agent.explain_instance()