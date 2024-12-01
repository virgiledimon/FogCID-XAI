import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class FCSpoofingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    ################################################################################################
    def __init__(self, r1, r2, l1, l2, n, xi_squared, m, omega, f0, L, area_size, nbr_legitimate_users, nbr_not_legitimate_users, nbr_receivers, actions):

        super(FCSpoofingEnv, self).__init__()

        # Paramètres de l'environnement

        self.r1 = r1  # Coût de réception des signaux illégaux
        self.r2 = r2  # Coût de refus des signaux légaux
        self.l1 = l1  # Gain de réception des signaux légaux
        self.l2 = l2  # Gain de refus des signaux illégaux

        self.n = n  # Taux de variation des gains de canal en dB
        self.xi_squared = xi_squared  # Gain de puissance moyen
        self.m = m  # Ratio des gains de canal de l'attaquant/légitime
        self.omega = omega  # SINR en dB
        self.f0 = f0  # Fréquence centrale
        self.area_size = area_size  # Taille de la zone de simulation en mètres
        self.L = L

        self.num_legitimate_users = nbr_legitimate_users  # Nombre d'attaquants
        self.num_not_legitimate_users = nbr_not_legitimate_users  # Nombre d'utilisateurs légitimes
        self.num_receivers = nbr_receivers
        
        self.Pk_list = self.generate_Pk()

        self.F = [f"F{i+1}" for i in range(nbr_receivers)]
        self.H = [f"H{i+1}" for i in range(nbr_legitimate_users)]  # Utilisateurs légitimes
        self.I = [f"I{i+1}" for i in range(nbr_not_legitimate_users)]  # Utilisateurs non-légitimes
        self.E = self.H + self.I

        self.receiver_coords = {f"F{i+1}": np.random.rand(2) * area_size for i in range(self.num_receivers)}
        
        # Créer une liste de valeurs lambda de 0.1 à 5.0 avec L valeurs
        self.actions = actions
        # self.action_space = spaces.Box(low=self.action_min_value, high=self.action_max_value, shape=(), dtype=np.float32) # seuil lambda
        self.action_space = spaces.Discrete(len(self.actions)) 
        
        # Création de l'espace d'observation (tous les couples possibles entre F et E)
        self.spaces = [(f, e) for f in self.F for e in self.E]
        # Espace d'observation défini par le nombre total de couples (discret)
        self.observation_space = spaces.Discrete(len(self.spaces))

        self.observation_space_n = len(self.E) * len(self.F)
        self.action_space_n = len(actions)

        # Initialisation des positions aléatoires des utilisateurs et attaquants
        self.reset()
    ################################################################################################

    ################################################################################################
    def reset(self, seed=None, options=None):
        
        # Initialisation des coordonnees des positions dans l'espace
        self.legitimate_coords = {f"H{i+1}": np.random.rand(2) * self.area_size for i in range(self.num_legitimate_users)}
        self.not_legitimate_coords = {f"I{i+1}": np.random.rand(2) * self.area_size for i in range(self.num_not_legitimate_users)}
        self.transmitter_coords = {**self.legitimate_coords, **self.not_legitimate_coords}

        self.receivers_list = self.F
        self.transmitters_list = self.E
        self.used_transmitters = set()

        # Selection aleatoire d'un recepteur
        self.current_receiver_id = np.random.choice(self.receivers_list)
        self.current_receiver_coords = self.receiver_coords[self.current_receiver_id]
        # Selection aleatoire d'un transmetteur initial
        self.current_transmitter_id = self.get_random_transmitter()
        # Definition de l'etat ie: (F1,H1)
        self.state = (self.current_receiver_id, self.current_transmitter_id)
        # Récupérer l'index de ce couple dans observation_space
        self.state_index = self.get_state_index(self.state)

        return int(self.state_index), {}
    ################################################################################################

    ################################################################################################
    def step(self, action):
        # Appliquer l'action (ici, ajuster le seuil lambda pour la détection)
        lambda_threshold = self.actions[action]
       
        # Vérifier si le transmetteur est légitime ou un attaquant
        if self.is_legitimate_transmitter():
            # Transmetteur légitime
            V_channel_transmitted = self.simulate_channel_gain()
            noise = np.random.normal(0, 0.1)  # Moyenne = 0, écart type = 0.1
            R_channel_received = V_channel_transmitted + noise  # Appliquer le bruit
            is_legit = True
        else:
            # Transmetteur illégitime (attaquant)
            V_channel_transmitted = self.simulate_channel_gain()
            V_channel_transmitted = V_channel_transmitted * self.m  # Appliquer le ratio 0.2
            R_channel_received = V_channel_transmitted + np.random.normal(0, 1)
            is_legit = False

        # Calculer la valeur L pour le test d'hypothèse
        L_value = float((np.linalg.norm(V_channel_transmitted - R_channel_received) ** 2) / (np.linalg.norm(R_channel_received) ** 2))
        
        detection = self.hypothesis_test(L_value, lambda_threshold)
        
        reward = self.get_reward(is_legit, detection)
        
        if len(self.used_transmitters) == len(self.transmitters_list):
            done = True
            next_state_index = self.observation_space_n - 1
            
        else:
            done = False
            next_state_index = self.move_to_next_state()

        truncated = False

        info = {
            "reward": reward , 
            "is_legit": bool(is_legit), 
            "L_value": L_value, 
            "detection": bool(detection), 
            "V_channel_transmitted": V_channel_transmitted, 
            "R_channel_received": R_channel_received
        }

        return next_state_index, done, truncated, info
    ################################################################################################

    ################################################################################################
    def render(self, mode='human'):
        plt.figure(figsize=(6, 6))
    
        # Affichage des récepteurs en bleu avec leurs noms
        for name, coords in self.receiver_coords.items():
            plt.scatter(coords[0], coords[1], color='blue', label='Récepteur' if name == list(self.receiver_coords.keys())[0] else "")
            plt.text(coords[0], coords[1], name, fontsize=9, ha='right')  # Ajouter le nom à côté du point
        
        # Affichage des transmetteurs légitimes en vert avec leurs noms
        for name, coords in self.legitimate_coords.items():
            plt.scatter(coords[0], coords[1], color='green', label='Transmetteur légitime' if name == list(self.legitimate_coords.keys())[0] else "")
            plt.text(coords[0], coords[1], name, fontsize=9, ha='right')  # Ajouter le nom à côté du point
        
        # Affichage des transmetteurs non-légitimes en rouge avec leurs noms
        for name, coords in self.not_legitimate_coords.items():
            plt.scatter(coords[0], coords[1], color='red', label='Transmetteur non-légitime' if name == list(self.not_legitimate_coords.keys())[0] else "")
            plt.text(coords[0], coords[1], name, fontsize=9, ha='right')  # Ajouter le nom à côté du point
        
        # Marquer le récepteur sélectionné avec une couleur différente (jaune) et son nom
        plt.scatter(self.current_receiver_coords[0], self.current_receiver_coords[1], color='yellow', s=200, label='Récepteur sélectionné')
        plt.text(self.current_receiver_coords[0], self.current_receiver_coords[1], self.current_receiver_id, fontsize=9, ha='right')
        
        # Légende, grille et limites du graphique
        plt.legend()
        plt.grid(True)
        plt.xlim(0, self.area_size)
        plt.ylim(0, self.area_size)
        plt.title("Visualisation des positions avec les noms des transmetteurs et récepteurs")
        plt.show()
    ################################################################################################

    ################################################################################################
    def close(self):
        pass
    ################################################################################################

    ################################################################################################
    def get_state_index(self, state_tuple):
        return self.spaces.index(state_tuple)
    ################################################################################################

    ################################################################################################
    def move_to_next_state(self):
        self.current_transmitter_id = self.get_random_transmitter()
        self.state = (self.current_receiver_id, self.current_transmitter_id)
        self.state_index = self.get_state_index(self.state)
        return self.state_index

    ################################################################################################

    ################################################################################################
    def is_legitimate_transmitter(self):
        if self.current_transmitter_id in self.H:
            return True  # C'est un utilisateur légitime
        elif self.current_transmitter_id in self.I:
            return False  # C'est un utilisateur non-légitime
    ################################################################################################

    ################################################################################################
    def get_random_transmitter(self):
        transmitter_id = np.random.choice([t for t in self.transmitters_list if t not in self.used_transmitters])
        self.used_transmitters.add(transmitter_id)
        return transmitter_id
    ################################################################################################

    ################################################################################################
    # Fonction pour générer la probabilité Pk basée sur le nombre d'attaquants et d'utilisateurs légitimes
    def generate_Pk(self):
        total_transmitters = self.num_legitimate_users + self.num_not_legitimate_users
        Pk = self.num_not_legitimate_users / total_transmitters
        if self.num_not_legitimate_users > 0:
            Pk_per_attacker = Pk / self.num_not_legitimate_users
        else:
            Pk_per_attacker = 0
        return [Pk_per_attacker] * self.num_not_legitimate_users
    ################################################################################################

    ################################################################################################
    def get_reward(self, is_legit, detection):

        # Déterminer l'reward en fonction des cas (comme expliqué précédemment)
        if detection and is_legit:
            # Signal légitime correctement accepté
            reward = self.calculate_reward(self.l1, 0, 0, 0)
        elif not detection and not is_legit:
            # Signal illégitime correctement rejeté
            reward = self.calculate_reward(0, self.l2, 0, 0)
        elif detection and not is_legit:
            # Signal illégitime incorrectement accepté
            reward = self.calculate_reward(0, 0, self.r1, 0)
        else:
            # Signal légitime incorrectement rejeté
            reward = self.calculate_reward( 0, 0, 0, self.r2)

        return reward
    ################################################################################################

    ################################################################################################
    # Méthode pour calculer la recompense
    def calculate_reward(self, l1, l2, r1, r2):

        sum_Pk = np.sum(self.Pk_list)

        reward = (l2 - l1) * sum_Pk \
                  - (l2 + r1) * self.P2_lambda * sum_Pk \
                  - (l1 + r2) * self.P1_lambda * (1 - sum_Pk) \
                  + l1

        return reward
    ################################################################################################

    ################################################################################################
    # Fonction pour générer des gains de canal aléatoires suivant une distribution normale
    def simulate_channel_gain(self):
        return np.random.normal(0, 1)
    ################################################################################################

    ################################################################################################
    # Fonction pour appliquer l'hypothèse de test
    def hypothesis_test(self, L, lambda_threshold):
        return L < lambda_threshold
    ################################################################################################

    ################################################################################################
    def set_probabilities(self, P1, P2):
        self.P1_lambda = P1
        self.P2_lambda = P2
    ################################################################################################
