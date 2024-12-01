import numpy as np
import scipy.stats as stats
import gymnasium as gym
import matplotlib.pyplot as plt
import gym_fc_impersonation_detection

class DoubleSarsaAgent:

    ################################################################################################
    def __init__(self, db_agent, r1=4, r2=2, l1=6, l2=9, mu=0.9, gamma=0.6, epsilon=0.5, n=3, xi_squared=5, m=0.2, omega=20, f0=2.4, T=20, N=30, area_size=50, nbr_legitimate_users=10, nbr_not_legitimate_users=5, nbr_receivers=5) :

        # Building the environment
        self.T = T  # Nombre d'etapes
        self.N = N  # Nombre épisodes
        self.L = 50
        self.gamma = gamma  # Taux de remise
        self.epsilon = epsilon  # Stratégie épsilon-gourmande
        self.mu = mu  # Efficacité d'apprentissage
        self.omega = omega
        self.n = n
        self.m = m
        self.xi_squared = xi_squared
        self.action_max_value = 2.0
        self.action_min_value = 0.1
        self.actions = [self.action_min_value + (self.action_max_value / (self.L - 1)) * i for i in range(self.L+1)]
        self.env = gym.make("FCImpersonationDetection-v0", r1=r1, r2=r2, l1=l1, l2=l2, n=n, xi_squared=xi_squared, m=m, omega=omega, f0=f0, L=self.L, area_size=area_size, nbr_legitimate_users=nbr_legitimate_users, nbr_not_legitimate_users=nbr_not_legitimate_users, nbr_receivers=nbr_receivers, actions=self.actions)

        # Initialisation des Q-tables
        self.QA = np.zeros((self.env.get_wrapper_attr('observation_space_n'), self.env.get_wrapper_attr('action_space_n')))
        self.QB = np.zeros((self.env.get_wrapper_attr('observation_space_n'), self.env.get_wrapper_attr('action_space_n')))
        self.pi = np.zeros(self.env.get_wrapper_attr('action_space_n'))

        self.db_agent = db_agent  # Agent PostgreSQL pour stocker les données
    ################################################################################################

    ################################################################################################
    # Function to choose the next action
    def choose_action(self, state):
        lambda_threshold = self.action_min_value
        if np.random.uniform(0, 1) < self.epsilon:
            lambda_threshold = self.env.action_space.sample()
        else:
            lambda_threshold = np.argmax(self.pi)
        return lambda_threshold
    ################################################################################################

    ################################################################################################
    # Fonction pour calculer les probabilités P1 et P2
    def calculate_p1_p2(self, lambda_threshold):
        P1_lambda = 1 - (self.calculate_fx((2 * lambda_threshold * self.omega) / (2 * (self.omega ** 2) + self.n * self.omega * self.xi_squared)))
        P2_lambda = self.calculate_fx((2 * lambda_threshold * self.omega) / (2 * (self.omega ** 2) + (1 + self.m) * self.omega * self.xi_squared))
        return P1_lambda, P2_lambda
    ################################################################################################

    ################################################################################################
    # Fonction pour calculer la fonction cumulative distribution F_X2
    def calculate_fx(self, value):
        return stats.chi2.cdf(value, 2)
    ################################################################################################

    ################################################################################################
    def update_q_tables(self, state, lambda_threshold, utility, next_state, lambda_threshold_prime):
        # Calcul des valeurs lambda optimales pour Q^A et Q^B
        lambda_A_prime = np.argmax(self.QA[next_state, :])
        lambda_B_prime = np.argmax(self.QB[next_state, :])

        # Mise à jour des Q-tables
        if np.random.rand() < 0.5:
            # Mise à jour de Q^A
            self.QA[state, lambda_threshold] = (1 - self.mu) * self.QA[state, lambda_threshold] + \
                self.mu * (utility + self.gamma * self.QB[next_state, lambda_A_prime])
        else:
            # Mise à jour de Q^B
            self.QB[state, lambda_threshold] = (1 - self.mu) * self.QB[state, lambda_threshold] + \
                self.mu * (utility + self.gamma * self.QA[next_state, lambda_B_prime])
    ################################################################################################

    ################################################################################################
    def run_simulation(self):

        FAR_list, MDR_list, AER_list, utility_list = [], [], [], []

        for episode in range(self.N):

            state, info = self.env.reset()

            t, far_episode, mdr_episode, aer_episode, utility_episode = 0, 0, 0, 0, 0
            decisions_to_insert = []  # Liste pour cumuler les décisions de cet épisode

            lambda_threshold = self.choose_action(state)

            self.env.render()

            while t < self.T:

                P1, P2 = self.calculate_p1_p2(lambda_threshold)

                unwrapped_env = self.env.unwrapped

                try:
                    unwrapped_env.set_probabilities(P1, P2)
                except AttributeError:
                    print("Error: 'FCImpersonationDetection-v0' environment does not have a 'set_probabilities' method.")

                # Take the action and observe the result
                next_state, done, truncated, step_training_data = self.env.step(lambda_threshold)

                step, is_legit, reward, L_value, detection, v_channel_vector, r_channel_record = t + 1, step_training_data['is_legit'], step_training_data['reward'], step_training_data['L_value'], step_training_data['detection'], step_training_data['V_channel_transmitted'], step_training_data['R_channel_received']

                # Choosing the next action
                lambda_threshold_prime = self.choose_action(next_state)

                #print(L_value, lambda_threshold, reward, v_channel_vector, r_channel_record, is_legit, detection)

                # Update the Q-value
                self.update_q_tables(state, lambda_threshold, reward, next_state, lambda_threshold_prime)

                pi_st = np.max((self.QA[state, :] + self.QB[state, :]) / 2)
                self.pi[lambda_threshold] = pi_st

                # print(
                #     episode, "episode",
                #     step,
                #     state,
                #     self.actions[lambda_threshold],
                #     is_legit,
                #     L_value,
                #     v_channel_vector,
                #     r_channel_record, "r_channel_record",
                #     detection,
                #     float(reward),
                #     float(P1),
                #     float(P2),
                #     float(P1+P2), "aer",
                #     float(np.max(self.QA[state, :])),
                #     float(np.max(self.QB[state, :])),
                #     float(pi_st)
                # )

                decisions_to_insert.append((
                    episode+1,
                    step,
                    state,
                    self.actions[lambda_threshold],
                    is_legit,
                    L_value,
                    v_channel_vector,
                    r_channel_record,
                    detection,
                    float(reward),
                    float(P1),
                    float(P2),
                    float(P1+P2),
                    float(np.max(self.QA[state, :])),
                    float(np.max(self.QB[state, :])),
                    float(pi_st)
                ))

                state = next_state
                lambda_threshold = lambda_threshold_prime

                # Accumulation des métriques pour l'épisode
                far_episode += P1
                mdr_episode += P2
                aer_episode += (P1+P2)
                utility_episode += reward

                if done:
                    break
                else:
                  t += 1

            # Affichage des résultats
            print(f"Épisode {episode + 1}/{self.N} | FAR: {far_episode/t:.4f}, MDR: {mdr_episode/t:.4f}, AER: {aer_episode/t:.4f}, Utility: {utility_episode/t:.4f}")
            print("\n")

            # Une fois tous les créneaux temporels terminés, insérer les décisions en une seule fois
            self.db_agent.insert_training_steps_data_bulk(decisions_to_insert)

            # Stocker les moyennes pour l'analyse finale
            FAR_list.append(far_episode / t)
            MDR_list.append(mdr_episode / t)
            AER_list.append(aer_episode / t)
            utility_list.append(utility_episode / t)

        self.lambda_star = self.actions[np.argmax(self.pi)]

        return np.mean(FAR_list), np.mean(MDR_list), np.mean(AER_list), np.mean(utility_list), self.lambda_star
    ################################################################################################

    ################################################################################################
    # Fonction pour exécuter la simulation sur différentes valeurs de SINR (ω)
    def run(self):
        print(f"Simulation pour ω = 10 dB")
        test_FAR, test_MDR, test_AER, test_utility, test_threshold = self.run_simulation()
        print(f"Résumé pour ω = 10 dB -> FAR moyen: {test_FAR:.4f}, MDR moyen: {test_MDR:.4f}, AER moyen: {test_AER:.4f}, Utility moyenne: {test_utility:.4f}, Threshold: {test_threshold:.4f}\n")
    ################################################################################################

    ################################################################################################
