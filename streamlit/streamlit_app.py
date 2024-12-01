import streamlit as st
import pandas as pd
import time
from detection_agents.DoubleSarsaAgent import DoubleSarsaAgent
from explicability_agents.SHAPAgent import SHAPAgent
from explicability_agents.LIMEAgent import LIMEAgent
from explicability_agents.PFIAgent import PFIAgent
from data.PostgreSQLAgent import PostgreSQLAgent
from interpretability_agents.InterpretabilityAgent import InterpretabilityAgent
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import shap
from streamlit_shap import st_shap
import dtreeviz
import graphviz as graphviz
from sklearn import tree
import numpy as np
import os
from sshtunnel import SSHTunnelForwarder

st.set_page_config(
    page_title="FogCID-XAI",
    layout = "centered"
)

# env variables
bd_over_ssh = st.secrets['STREAMLIT_DB_OVER_SSH'] 

# Secrets variables
secret_db_name = st.secrets["db_name"]
secret_db_user = st.secrets["db_user"]
secret_db_password = st.secrets["db_password"]
secret_db_port = st.secrets["db_port"]

def display_global_shap_plots(shapley_values, dataframe_test, features):
    ssp_container1 = st.container()
    with ssp_container1:
        st_shap(shap.summary_plot(shapley_values, dataframe_test, plot_size=[8,2]))
    ssp_container2 = st.container()
    with ssp_container2:
        st_shap(shap.summary_plot(shapley_values, dataframe_test, plot_type="bar", plot_size=[8,2]))
    sdp_container1 = st.container()
    with sdp_container1:
        st_shap(shap.dependence_plot(features[0], shapley_values, dataframe_test, interaction_index=features[1]), height=300, width=700)
    sdp_container2 = st.container()
    with sdp_container2:
        st_shap(shap.dependence_plot(features[1], shapley_values, dataframe_test, interaction_index=features[0]), height=300, width=700)
    

def simulate(database_agent, episodes_number, ml_algorithm, records_number):
    interpretability_agent = InterpretabilityAgent()

    # Initialisation des agents 
    detection_agent = DoubleSarsaAgent(db_agent=database_agent, N=episodes_number) if ml_algorithm == "Double SARSA" else None

    # Exécution de l'agent de détection et collecte des données
    detection_agent.run_simulation()  # Lancement de la simulation

    # Récupérer les données de décisions via l'agent PostgreSQL
    decision_data = database_agent.fetch_decisions_data(records_number)
    decision_data_allfields = database_agent.fetch_decisions_data_allfields(records_number)

    # Séparer les caractéristiques (lambda_threshold, l_value) et la cible (decision_made)
    X = decision_data[:, [0, 1]]  # Caractéristiques : lambda_threshold, l_value
    y = decision_data[:, 2]  # Cible : decision_made

    # Séparation des données en ensembles d'entraînement et de test (50/50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Normalisation des données
    X_train_normalized, X_test_normalized = interpretability_agent.normalize_decision_data(X_train, X_test)

    ############################## DETETECTION SEQUENCES DATA
    st.header("1. Detection sequences")
    st.divider()

    st.subheader("1.1 List of observation states")
    with st.expander("See"):
        observations = detection_agent.env.get_wrapper_attr('spaces')
        st.write(observations)

    st.subheader("1.2 Data preview")
    detection_df = pd.DataFrame(
        decision_data_allfields, 
        columns=["episode", "step", "state", "lambda_threshold", "L_value", "is_legit", "decision_made", "v_channel_vector", "r_channel_record", "reward", "far", "mdr", "aer", "q_value_a", "q_value_b", "politique_pi"]
    )
    eventDataset = st.dataframe(
        detection_df, 
        use_container_width=True, 
        key="data", 
        #on_select="rerun", 
        #selection_mode=["multi-row", "multi-column"]
    )
    st.caption(f"Number of records: {records_number}")

    ############################## INTERPRETABILITY
    st.header("2. Interpretability")
    st.divider()
    st.markdown("### Decision Tree")
    interpretability_agent.train_decision_model(X_train_normalized, y_train)
    accuracy = interpretability_agent.evaluate_model(X_test_normalized, y_test)
    decision_tree_container = st.container()
    with decision_tree_container:
        dot_data = tree.export_graphviz(interpretability_agent.model_decision, out_file=None, filled=True, feature_names=['lambda_threshold', 'l_value'], class_names=['Reject', 'Accept'])
        graph = graphviz.Source(dot_data, format="png")
        graph
        # fig = plt.figure(figsize=(20, 10))
        # plot_tree(interpretability_agent.model_decision, filled=True, feature_names=['lambda_threshold', 'l_value'], class_names=['Reject', 'Accept'])
        # st.pyplot(fig)
        # plt.clf()
        st.caption("Accuracy for Decision Tree  " + str(round(accuracy,2)))
    
    ############################## EXPLICABILITY
    st.header("3. Explicability")
    st.divider()

    # # Pour SHAP : Affiche les valeurs de SHAP sous forme de graphique
    st.subheader("3.1 SHapley Additive exPlanations")
    shap_agent = SHAPAgent(interpretability_agent.model_decision, X_test_normalized)
    shapley_values, features_names, df_X_test = shap_agent.explain_model() 
    # Explisubheadercability global
    display_global_shap_plots(shapley_values, df_X_test, features_names)
    # Explicability local
    expected_local_values, shap_local_values, shap_local_columns = shap_agent.explain_instance()
    st_shap(shap.force_plot(expected_local_values, shap_local_values, feature_names=shap_local_columns, link="logit", matplotlib=False, show=True, figsize=(9,4)))

    st.subheader("3.2 Local Interpretable Model-Agnostic Explanations")
    lime_agent = LIMEAgent(model=interpretability_agent.model_decision, normalized_test_data=X_test_normalized)
    X_local_instance = lime_agent.X_lime_test[0]
    explainer = lime_agent.explain_instance(X_instance=X_local_instance)
    
    st.pyplot(explainer.as_pyplot_figure())
    plt.clf()
    components.html(explainer.as_html(), height=300, width=700)
    
    st.subheader("3.3 Permutation Feature Importance")
    pfi_agent = PFIAgent(interpretability_agent.model_decision, X_test_normalized, y_test)
    importances = pfi_agent.compute_importance()
    st.write("Features importances:", importances)
    feature_names, indices = pfi_agent.plot_importance(importances)
    fig = plt.figure(figsize=(10, 6))
    plt.title("Features importances")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45)
    plt.xlim([-1, len(importances)])
    plt.ylabel("Importance average")
    plt.xlabel("Features")
    plt.grid(axis='y')
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

def main():
    st.title("FogCID-XAI")
    st.subheader("Explainable AI approach for impersonation attack detection in fog computing")
    st.write("This project focuses on enhancing the explainability of machine learning models for detecting impersonation attacks in fog computing. \
             The machine learning model used for the simulation is Double SARSA, a reinforcement learning algorithm. \
             While three explainability approaches were identified, this application specifically emphasizes explaining the agent's decisions to accept or reject signals. \
             It integrates decision trees alongside techniques such as SHAP, LIME, and PFI to provide insights into the model's decision-making process. \
             The goal is to ensure security, transparency, and interpretability in critical IoT environments.")
    st.divider()

    st.sidebar.markdown("# Simulation Configuration")

    tunnel = None

    # Formulaire de choix de simulation
    with st.sidebar:
        with st.form("input_params_form", clear_on_submit=False):
            st.write("Update Parameters")
            algo = st.selectbox("Detection Agent", ["Double SARSA"])
            episode_nbr = st.number_input("Sequences episodes number", min_value=30, value=30)
            dataset_rows_nbr = st.number_input("Dataset records number", min_value=0, value=episode_nbr*20, max_value=episode_nbr*20)
            receivers_number = st.number_input("Number of receivers devices", min_value=0, value=5)
            leg_transmitters_number = st.number_input("Number of legitimate transmitters", min_value=0, value=10)
            illeg_transmitter_number = st.number_input("Number of illegitimate transmitters", min_value=0, value=5)
            submitted = st.form_submit_button("Rerun simulation")
    
    if bd_over_ssh == "YES":
        # Create an SSH tunnel
        tunnel = SSHTunnelForwarder(
            ('premium226.web-hosting.com', 21098),
            ssh_username='virgvakl',
            ssh_password='Ibelievein@2024',
            remote_bind_address=('127.0.0.1', 5432),
            local_bind_address=('127.0.0.1', 5555), # could be any available port
        )
        # Start the tunnel
        tunnel.start()

    db_agent = PostgreSQLAgent(db_name='virgvakl_fogcid_xai', user='virgvakl_fogcid_xai_user', password='naThisIsFogProject2024', port=5555)

    # By default when the page is loaded, we take defaults values and run the simulation
    simulate(db_agent, episode_nbr, algo, dataset_rows_nbr)

    # Lorsque le formulaire est soumis (Rerunning)
    if submitted:
        simulate(db_agent, episode_nbr, algo, dataset_rows_nbr)
    
    db_agent.close()
    if tunnel is not None:
        tunnel.stop(force=True)
        tunnel.close()

if __name__ == "__main__":
    main()