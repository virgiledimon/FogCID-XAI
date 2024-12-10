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
    

def simulate(episodes_number, ml_algorithm, records_number, test_data_size, instance_idx, receivers_nbr, leg_transm_nbr, illeg_transm_nbr):
    
    with st.spinner('In progress...'):

        tunnel = None
        database_agent = None

        #try:

        if st.secrets['STREAMLIT_DB_OVER_SSH'] == "YES":
            # Create an SSH tunnel
            tunnel = SSHTunnelForwarder(
                (st.secrets["db_ssh_host"], st.secrets["db_ssh_port"]),
                ssh_username=st.secrets["db_ssh_username"],
                ssh_password=st.secrets["db_ssh_password"],
                remote_bind_address=(st.secrets["db_remote_bind_address"], st.secrets["db_remote_bind_port"]),
                local_bind_address=(st.secrets["db_local_bind_address"], st.secrets["db_local_bind_port"]), 
            )

            # Start the tunnel
            try:
                tunnel.start()
            except:
                tunnel.stop(force=True)
                tunnel.close()
                st.error("Connection to ssh tunnel for database failed. Please try later")
                return False
        
        database_agent = PostgreSQLAgent(db_name=st.secrets["db_name"], user=st.secrets["db_user"], password=st.secrets["db_password"], host=st.secrets["db_host"], port=st.secrets["db_port"])
        
        interpretability_agent = InterpretabilityAgent()

        # Initialisation des agents 
        detection_agent = DoubleSarsaAgent(db_agent=database_agent, N=episodes_number, nbr_legitimate_users=leg_transm_nbr, nbr_not_legitimate_users=illeg_transm_nbr, nbr_receivers=receivers_nbr) if ml_algorithm == "Double SARSA" else None

        # Exécution de l'agent de détection et collecte des données
        #detection_agent.run_simulation()  # Lancement de la simulation

        # Récupérer les données de décisions via l'agent PostgreSQL
        decision_data = database_agent.fetch_decisions_data(records_number)
        
        # Séparer les caractéristiques (lambda_threshold, l_value) et la cible (decision_made)
        X = decision_data[:, [0, 1]]  # Caractéristiques : lambda_threshold, l_value
        y = decision_data[:, 2]  # Cible : decision_made

        # Séparation des données en ensembles d'entraînement et de test (50/50)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_size, random_state=42)

        # Normalisation des données
        X_train_normalized, X_test_normalized = interpretability_agent.normalize_decision_data(X_train, X_test)

        ############################## DETETECTION SEQUENCES DATA
        st.markdown("### 1. Detection sequences")

        st.markdown("#### 1.1 List of observation states")

        with st.expander("See"):
            observations = detection_agent.env.get_wrapper_attr('spaces')
            st.write(observations)

        st.markdown("#### 1.2 Data preview")
        
        st.write("At each interaction step, the agent gathers detailed information, such as the state, selected threshold (λ), signal legitimacy, channel vectors, decisions made, rewards, and performance metrics (FAR, MDR, AER). The collected data is stored in a relational table optimized using TimescaleDB, ensuring efficient temporal data management. This approach allows for detailed performance analysis and traceability, supporting the optimization of the detection agent's strategies.")
        decision_data_allfields = database_agent.fetch_decisions_data_allfields(records_number)
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

        with st.expander("See fields description"):
            # Table of field descriptions
            dataStructure = {
                "Field": [
                    "id", "episode", "step", "state", "lambda_threshold", "is_legit", "L_value", 
                    "v_channel_vector", "r_channel_record", "decision_made", "reward", 
                    "far", "mdr", "aer", "q_value_a", "q_value_b", "politique_pi", "timestamp"
                ],
                "Description": [
                    "Unique identifier (primary key).",
                    "Episode number.",
                    "Step number within the episode.",
                    "Identifier of the state.",
                    "Detection threshold (λ) selected for the state.",
                    "Status of the signal (legitimate or attacker).",
                    "Normalized Euclidean distance between transmitted and received channel vectors.",
                    "Channel vector transmitted by the emitter (legitimate or attacker).",
                    "Channel vector received after transmission and channel perturbation.",
                    "Decision on the signal (accepted or rejected).",
                    "Reward assigned based on detection outcome.",
                    "False Alarm Rate (FAR).",
                    "Missed Detection Rate (MDR).",
                    "Average Error Rate (AER).",
                    "Updated value in Q-table QA.",
                    "Updated value in Q-table QB.",
                    "Average maximum value of QA + QB.",
                    "created_at and updated_at."
                ]
            }
            df = pd.DataFrame(dataStructure)
            st.dataframe(df, use_container_width=True)
        ############################## INTERPRETABILITY
        st.markdown("### 2. Interpretability")

        st.divider()

        st.markdown("The Double SARSA algorithm effectively detects attacks but lacks transparency in its decision-making. By converting its processes into a decision tree model, key actions like threshold selection and signal evaluation become interpretable. This approach fosters user trust, simplifies performance analysis, and ensures transparency in the detection process.")

        st.markdown("##### Decision Tree")
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

        with st.expander("See description"):
            st.markdown("""
                ##### Key Elements of a Decision Tree

                When visualizing a decision tree, each of node provides key information about the decision-making process. Here's a breakdown of the elements we might encounter:

                ###### **Feature Split (e.g., `lambda_threshold <= 0.459`)**
                This represents the condition used to split the data at this node. The node splits based on whether the feature `lambda_threshold` is less than or equal to `0.459`.

                ###### **Gini (e.g., `gini = 0.085`)**
                The Gini Impurity measures the purity of the node:
                - **Gini = 0**: The node is perfectly pure (all samples belong to one class).
                - **Higher Gini**: Indicates more mixed classes in the node.

                ###### **Samples (e.g., `samples = 45`)**
                Indicates the number of data points present in this node. For this instance, `samples = 45` means 45 data points reach this node.

                ###### **Values (e.g., `value = [43, 2]`)**
                Represents the class distribution of the samples:
                `value = [43, 2]`: 43 samples belong to one class, and 2 belong to the other.

                ###### **Class (e.g., `class = Reject`)**
                The predicted class for this node:
                - Based on the majority class in `value`. For example:
                - If `value = [43, 2]`, the node predicts the class "Reject".
                        
                ###### Node colors
                Node colors reflect the dominance of a class:
                - Nodes with a single class are solid-colored.
                - Mixed nodes show blended colors, proportional to the class distribution.

                ---
                """
            )
            st.markdown("##### Example breakdown")
            st.markdown("""
                ###### Node 1
                The example below indicates that the most samples at this node (43 out of 45) belong to the "Reject" class. The node is highly pure so has low Gini value.
            """)
            st.code(''' 
                lambda_threshold <= 0.459
                Gini: 0.085
                Samples: 45
                Values: [43, 2]
                Class: Reject
            ''', language=None)
            st.markdown("""
                ###### Node 2
                The node below is less pure (Gini = 0.484), with 7 samples in the "Reject" class and 10 in the "Accept" class. The predicted class is "Accept", based on the majority in `value`.
            """)
            st.code(''' 
                l_value <= -0.067
                Gini: 0.484
                Samples: 17
                Values: [7, 10]
                Class: Accept
            ''', language=None)
        ############################## EXPLICABILITY
        st.markdown("### 3. Explainability")

        st.divider()

        # # Pour SHAP : Affiche les valeurs de SHAP sous forme de graphique
        st.markdown("#### 3.1 SHapley Additive exPlanations")

        shap_agent = SHAPAgent(interpretability_agent.model_decision, X_test_normalized)
        shapley_values, features_names, df_X_test = shap_agent.explain_model() 
        # Explisubheadercability global
        display_global_shap_plots(shapley_values, df_X_test, features_names)
        # Explicability local
        expected_local_values, shap_local_values, shap_local_columns = shap_agent.explain_instance(instance_idx)
        st_shap(shap.force_plot(expected_local_values, shap_local_values, feature_names=shap_local_columns, link="logit", matplotlib=False, show=True, figsize=(9,4)))

        st.markdown("#### 3.2 Local Interpretable Model-Agnostic Explanations")

        lime_agent = LIMEAgent(model=interpretability_agent.model_decision, normalized_test_data=X_test_normalized)
        X_local_instance = lime_agent.X_lime_test[instance_idx]
        explainer = lime_agent.explain_instance(X_instance=X_local_instance)
        
        st.pyplot(explainer.as_pyplot_figure())
        plt.clf()
        components.html(explainer.as_html(), height=300, width=700)
        
        st.markdown("#### 3.3 Permutation Feature Importance")

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

        # except:
        #     st.error("An exception occurred. Please try later")

    if database_agent is not None:
        database_agent.close()
    if tunnel is not None:
        tunnel.stop(force=True)
        tunnel.close()

def main():
    st.title("FogCID-XAI")
    st.markdown("#### Explainable AI approach for impersonation attack detection in fog computing")
    st.write("This project focuses on enhancing the explainability of machine learning models for detecting impersonation attacks in fog computing. \
             The machine learning model used for the simulation is Double SARSA, a reinforcement learning algorithm. \
             While several explainability aspects were identified, this application specifically emphasizes explaining the agent's decisions to accept or reject signals. \
             It integrates decision trees alongside techniques such as SHAP, LIME, and PFI to provide insights into the model's decision-making process. \
             The goal is to ensure security, transparency, and interpretability in critical IoT environments.")
    
    st.markdown("##### Explainable aspects of our RL detection agent")

    st.markdown(
        """
        - **Dynamic threshold selection (λ):**  
        The Double SARSA algorithm adjusts the threshold dynamically using Q-values, which can be explained using methods like SHAP and LIME  

        - **Signals acceptance decisions:**  
        Decisions to accept or reject signals depend on the dynamic threshold (λ) and the normalized Euclidean distance  

        - **Performance optimization:**  
        The algorithm minimizes detection errors (FAR, MDR, AER) and maximizes utility through optimal threshold selection
        """
    )

    st.divider()

    st.subheader("Explanation of acceptance or rejection decisions made by our RL detection agent")

    st.markdown(
        """
        | **Environment**                 | Detection of attacks in fog computing                                          |
        |---------------------------------|---------------------------------------------------------------------------------|
        | **Agent**                       | Double SARSA                                                                   |
        | **Type of Explanation**         | Local and Global [Explanation provided by SHAP, LIME, PFI]                    |
        | **What Needs to be Explained**  | Decisions (acceptance or rejection of signals) made by the RL agent |
        | **Explanation Process**         | Training an intrinsically interpretable decision tree based on collected data. Features are then extracted and analyzed to provide explanations to the target audience. |
        | **Decision Process**            | MDP (Markov Decision Process) and policy updates using Double SARSA           |
        | **Target Audience**             | Administrators and cybersecurity experts                                       |
        | **Features**                    | Detection threshold (λ), normalized Euclidean distance (L) of channel gains between the receiver and the transmitter |
        """
    )

    #####################################################################################################
    
    st.sidebar.markdown("# Simulation Configuration")

    # Formulaire de choix de simulation
    with st.sidebar:
        with st.form("input_params_form", clear_on_submit=False):
            st.write("Update Parameters")
            algo = st.selectbox("Detection Agent", ["Double SARSA"])
            episode_nbr = st.number_input("Sequences episodes number", min_value=30, value=30)
            dataset_rows_nbr = st.number_input("Dataset records number", min_value=0, value=episode_nbr*20, max_value=episode_nbr*20)
            test_size = st.number_input("Test size data ratio (%)", min_value=30, value=50, max_value=70)
            receivers_number = st.number_input("Number of receivers devices", min_value=0, value=5)
            leg_transmitters_number = st.number_input("Number of legitimate transmitters", min_value=0, value=10)
            illeg_transmitter_number = st.number_input("Number of illegitimate transmitters", min_value=0, value=5)
            local_instance_idx = st.number_input("Local explicability instance idx", min_value=0, value=1, max_value=int(dataset_rows_nbr*(test_size/100)))
            submitted = st.form_submit_button("Rerun simulation")
        
    #####################################################################################################

    # Lorsque le formulaire est soumis (Rerunning)
    if submitted:
        simulate(
            episode_nbr, 
            algo, 
            dataset_rows_nbr, 
            test_size/100, 
            local_instance_idx, 
            receivers_number, 
            leg_transmitters_number, 
            illeg_transmitter_number
        )
    else:
        # By default when the page is loaded, we take defaults values and run the simulation
        simulate(
            episode_nbr, 
            algo, 
            dataset_rows_nbr, 
            test_size/100, 
            local_instance_idx, 
            receivers_number, 
            leg_transmitters_number, 
            illeg_transmitter_number
        )
        
if __name__ == "__main__":
    main()