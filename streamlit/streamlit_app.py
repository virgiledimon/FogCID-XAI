import streamlit as st
import pandas as pd
import time
from detection_agents.DoubleSarsaAgent import DoubleSarsaAgent  # à ajuster selon l'emplacement des agents
from explicability_agents.SHAPAgent import SHAPAgent
from explicability_agents.LIMEAgent import LIMEAgent
from explicability_agents.PFIAgent import PFIAgent
from data.PostgreSQLAgent import PostgreSQLAgent
from interpretability_agents.InterpretabilityAgent import InterpretabilityAgent
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import streamlit.components.v1 as components

st.set_page_config(
    page_title="FogCID-XAI",
    layout = "wide"
)

if __name__ == "__main__":

    st.title("FogCID-XAI")
    st.write("Explainable AI approach for impersonation attack detection in fog computing")
    st.divider()

    st.sidebar.markdown("# Simulation Form")

    # Formulaire de choix de simulation
    with st.sidebar:
        with st.form("input_params_form", clear_on_submit=False):
            st.write("Faites un choix de simulation")
            algo = st.selectbox("Agent de détection", ["Double SARSA"])
            episode_nbr = st.number_input("Nombre d'épisodes", min_value=30)
            submitted = st.form_submit_button("Run simulation")

    db_agent = PostgreSQLAgent(db_name='virgvakl_fogcid_xai', user='virgvakl_fogcid_xai_user', password='naThisIsFogProject2024', port='5522')

    # Lorsque le formulaire est soumis
    if submitted:

        interpretability_agent = InterpretabilityAgent()

        # Initialisation des agents 
        detection_agent = DoubleSarsaAgent(db_agent=db_agent, N=episode_nbr) if algo == "Double SARSA" else None

        # Exécution de l'agent de détection et collecte des données
        #detection_agent.run_simulation()  # Lancement de la simulation

        # Récupérer les données de décisions via l'agent PostgreSQL
        decision_data = db_agent.fetch_decisions_data(600)
        decision_data_allfields = db_agent.fetch_decisions_data_allfields(600)

        # Séparer les caractéristiques (lambda_threshold, l_value) et la cible (decision_made)
        X = decision_data[:, [0, 1]]  # Caractéristiques : lambda_threshold, l_value
        y = decision_data[:, 2]  # Cible : decision_made

        # Séparation des données en ensembles d'entraînement et de test (50/50)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # Normalisation des données
        X_train_normalized, X_test_normalized = interpretability_agent.normalize_decision_data(X_train, X_test)

        ###### DETETECTION SEQUENCES DATA
        st.header("1. Detection sequences")
        st.divider()

        st.subheader("1.1 List of observation states")
        with st.expander("See"):
            observation_state_string = "|"
            observations = detection_agent.env.get_wrapper_attr('spaces')
            for i in range(len(observations)):
                observation_state_string += f" {i}: ({observations[i][0]},{observations[i][1]}) |"
            st.write(observation_state_string)

        st.subheader("1.2 Dataset")
        detection_df = pd.DataFrame(decision_data_allfields, columns=["episode", "step", "state", "lambda_threshold", "L_value", "is_legit", "decision_made", "v_channel_vector", "r_channel_record", "reward", "far", "mdr", "aer", "q_value_a", "q_value_b", "politique_pi"])
        st.write(detection_df)

        ###### INTERPRETABILITY
        st.header("2. Interpretability")
        st.divider()

        interpretability_agent.train_decision_model(X_train_normalized, y_train)
        accuracy = interpretability_agent.evaluate_model(X_test_normalized, y_test)

        if hasattr(interpretability_agent.model_decision, 'tree_'):
            fig = plt.figure(figsize=(20, 10))
            plot_tree(interpretability_agent.model_decision, filled=True,
                        feature_names=['lambda_threshold', 'l_value'],
                        class_names=['Reject', 'Accept'])
            st.pyplot(fig)
            st.caption("Accuracy for Decision Tree  " + str(round(accuracy,2)))
        else:
            st.write("Le modèle n'est pas encore entraîné. Entraînez d'abord le modèle avant de l'afficher.")

        ###### EXPLICABILITY
        st.header("3. Explicability")
        st.divider()

        # # Pour SHAP : Affiche les valeurs de SHAP sous forme de graphique
        st.markdown("### 3.1 SHapley Additive exPlanations")
        shap_agent = SHAPAgent(interpretability_agent.model_decision, X_test_normalized)
        shap_global_plots = shap_agent.explain_model()
        fig2 = plt.figure(figsize=(20, 10))
        
        st.pyplot(fig2)
        
        components.html(shap_agent.explain_instance(), height=None)

        st.markdown("### 3.2 Local Interpretable Model-Agnostic Explanations")
        

        st.markdown("### 3.3 Permutation Feature Importance")