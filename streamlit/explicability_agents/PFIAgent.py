from sklearn.inspection import permutation_importance

class PFIAgent:
    def __init__(self, model, db_agent):
        self.model = model
        self.db_agent = db_agent
        self.X_test = None  # Initialisation des données normalisées
        self.y_test = None  # Initialisation des étiquettes

    def get_normalized_decision_data(self):
        # Récupérer les données de décisions via l'agent PostgreSQL
        decision_data = self.db_agent.fetch_decisions_data_min()

        if decision_data is not None and len(decision_data) > 0:
            print("Données de décisions récupérées avec succès.")

            # Séparer les caractéristiques et la cible
            self.X_test = decision_data[:, [0, 1]]  # lambda_threshold, L_value
            self.y_test = decision_data[:, 2]  # decision_made

            # Normaliser les données
            scaler = StandardScaler()
            self.X_test = scaler.fit_transform(self.X_test)
        else:
            print("Erreur: Aucune donnée de décision disponible.")

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

        plt.figure(figsize=(10, 6))
        plt.title("Importance des caractéristiques")
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45)
        plt.xlim([-1, len(importances)])
        plt.ylabel("Importance moyenne")
        plt.xlabel("Caractéristiques")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

# Exemple d'utilisation
# if __name__ == "__main__":
#     # Supposons que db_agent est déjà initialisé et connecté à la base de données
#     db_agent = PostgreSQLAgent(db_name='', user='postgres', password='postgres')

#     # Créer l'agent PFI en utilisant le modèle déjà entraîné
#     pfi_agent = PFI_Agent(model=interpret_agent.model_decision, db_agent=db_agent)

#     # Récupérer et normaliser les données de décisions depuis la base
#     pfi_agent.get_normalized_decision_data()

#     # Calculer les importances des caractéristiques
#     importances = pfi_agent.compute_importance()
#     print("Importances des caractéristiques:", importances)

#     # Afficher le graphique des importances
#     pfi_agent.plot_importance(importances)