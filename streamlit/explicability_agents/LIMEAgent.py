import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
import numpy as np

class LIMEAgent:
    def __init__(self, model, db_agent):
        self.model = model
        self.db_agent = db_agent
        self.X_test = None  # Initialisation des données normalisées

    # Méthode pour récupérer et normaliser les données de décisions depuis la base
    def get_normalized_decision_data(self):
        # Récupérer les données via la méthode existante dans db_agent
        decision_data = self.db_agent.fetch_decisions_data_min()

        if decision_data is not None and len(decision_data) > 0:
            print("Données de décisions récupérées avec succès.")

            # Séparer les caractéristiques pertinentes (lambda_threshold et l_value)
            X_decision = decision_data[:, [0, 1]]
            print(X_decision[2])
            # Normaliser les données
            scaler = StandardScaler()
            X_decision_normalized = scaler.fit_transform(X_decision)
            self.X_test = X_decision_normalized  # Stocker les données normalisées dans la classe
        else:
            print("Erreur: Aucune donnée de décision disponible.")

    # Méthode pour expliquer une instance spécifique avec LIME
    def explain_instance(self, X_instance):
        if self.X_test is None:
            print("Les données ne sont pas encore disponibles. Veuillez appeler 'get_normalized_decision_data' avant.")
            return

        plt.style.use('default')  # Appliquer un style avec fond blanc

        # Initialisation de l'explainer LIME
        explainer = LimeTabularExplainer(
            self.X_test,
            feature_names=['lambda_threshold', 'L_value'],  # Les noms des caractéristiques
            class_names=['Reject', 'Accept'],
            mode='classification',
            verbose=True,
        )

        print("\n")

        # Expliquer l'instance donnée (qui doit être normalisée)
        exp = explainer.explain_instance(X_instance, self.model.predict_proba, num_features=2)
        exp.show_in_notebook(show_table=True)

        fig = exp.as_pyplot_figure()
        fig.savefig("lime_explanation_white_bg.png", bbox_inches='tight', facecolor='white')

        print("\n")

# Exemple d'utilisation
# if __name__ == "__main__":
#     # Supposons que db_agent est déjà initialisé et connecté à la base de données
#     db_agent = PostgreSQLAgent(db_name='', user='postgres', password='postgres')

#     # Créer l'agent LIME en utilisant le modèle déjà entraîné et la base de données
#     lime_agent = LIMEAgent(model=interpret_agent.model_decision, db_agent=db_agent)

#     # Récupérer et normaliser les données de décisions depuis la base
#     lime_agent.get_normalized_decision_data()

#     # Exemple de nouvelle donnée à expliquer (doit être normalisée avant utilisation)
#     X_instance_normalized = lime_agent.X_test[0]  # Utiliser la première instance des données normalisées
#     print(X_instance_normalized)
#     # Expliquer l'instance
#     lime_agent.explain_instance(X_instance_normalized)
