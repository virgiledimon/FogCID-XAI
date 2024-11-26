import psycopg2
import logging
from psycopg2 import extras
import numpy as np

class PostgreSQLAgent:
    def __init__(self, db_name, user, password, host='localhost', port='5432'):
        try:
            self.conn = psycopg2.connect(
                dbname=db_name,
                user=user,
                password=password,
                host=host,
                port=port
            )
            self.cursor = self.conn.cursor()
        except psycopg2.DatabaseError as e:
            logging.error(f"Erreur de connexion à la base de données : {e}")
            raise

    # Méthode pour insérer en masse des décisions
    def insert_training_steps_data_bulk(self, decisions_data):
          try:
              query = """
              INSERT INTO training_steps_data (episode, step, state, lambda_threshold, is_legit, L_value, v_channel_vector, r_channel_record, decision_made, reward, far, mdr, aer, q_value_a, q_value_b, politique_pi)
              VALUES %s;
              """
              extras.execute_values(self.cursor, query, decisions_data)
              self.conn.commit()
          except psycopg2.DatabaseError as e:
              logging.error(f"Erreur lors de l'insertion en masse des décisions dans training_steps_data : {e}")
              self.conn.rollback()

    # Nouvelle méthode pour récupérer les données de décisions avec jointure
    def fetch_decisions_data(self, nbr_records):
        try:
            query = """
            SELECT lambda_threshold, L_value, decision_made, v_channel_vector, r_channel_record, reward, is_legit
            FROM training_steps_data
            LIMIT %s
            """
            self.cursor.execute(query, [nbr_records])
            decision_data = self.cursor.fetchall()
            return np.array(decision_data)
        except psycopg2.DatabaseError as e:
            logging.error(f"Erreur lors de la récupération des données de décisions : {e}")
            self.conn.rollback()
            return None
    
    def fetch_decisions_data_allfields(self, nbr_records):
        try:
            query = """
            SELECT episode, step, state, lambda_threshold, L_value, is_legit, decision_made, v_channel_vector, r_channel_record, reward, far, mdr, aer, q_value_a, q_value_b, politique_pi
            FROM training_steps_data
            LIMIT %s
            """
            self.cursor.execute(query, [nbr_records])
            decision_data = self.cursor.fetchall()
            return np.array(decision_data)
        except psycopg2.DatabaseError as e:
            logging.error(f"Erreur lors de la récupération des données de décisions : {e}")
            self.conn.rollback()
            return None

    # Nouvelle méthode pour récupérer les données de décisions avec jointure
    def fetch_decisions_data_min(self):
        try:
            query = """
            SELECT lambda_threshold, L_value, decision_made, v_channel_vector, r_channel_record
            FROM training_steps_data
            LIMIT 200
            """
            self.cursor.execute(query)
            decision_data = self.cursor.fetchall()
            return np.array(decision_data)
        except psycopg2.DatabaseError as e:
            logging.error(f"Erreur lors de la récupération des données de décisions : {e}")
            self.conn.rollback()
            return None

    def close(self):
        try:
            self.cursor.close()
            self.conn.close()
        except psycopg2.DatabaseError as e:
            logging.error(f"Erreur lors de la fermeture de la connexion à la base de données : {e}")