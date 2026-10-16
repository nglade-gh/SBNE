import json
import sqlite3
import numpy as np
import hashlib
from jsonpatch import make_patch

# --- Fonctions utilitaires ---

def extract_network(sb_network):
    """Extraire states et weights depuis un objet SbNetwork."""
    return {
        "states": sb_network.states.tolist(),
        "weights": sb_network.weights.tolist()
    }

def compute_network_hash(network_dict):
    """Calculer un hash unique pour un réseau."""
    network_str = json.dumps(network_dict, sort_keys=True)
    return hashlib.sha256(network_str.encode('utf-8')).hexdigest()

def get_last_commit_id(cursor, agent_id):
    """Récupérer l'ID du dernier commit pour un agent."""
    cursor.execute("""
        SELECT id_commit
        FROM commits
        WHERE id_agent = ?
        ORDER BY iteration DESC
        LIMIT 1
    """, (agent_id,))
    result = cursor.fetchone()
    return result[0] if result else None

# --- Sauvegarde ---
def save_network_state(db_path, agent_id, parent_agent_id, sb_network, iteration, is_snapshot=False, message="Update"):
    """
    Sauvegarde l'état d'un agent et de son réseau neuronal dans la base de données.

    Args:
        db_path (str): Chemin vers la base de données SQLite (ex: "simulations/sim_20260709_140430/evolution.db").
        agent_id (str/int): ID unique de l'agent (ex: "20260709_140430_1" ou 1).
        parent_agent_id (str/int/None): ID du parent (None pour les agents initiaux).
        sb_network: Objet SbNetwork contenant states et weights.
        iteration (int): Itération actuelle de la simulation.
        is_snapshot (bool): True pour un snapshot complet, False pour une mise à jour incrémentale.
        message (str): Message descriptif (ex: "Agent initial", "Division").
    """
    # 1. Extrait les données du réseau
    current_network = extract_network(sb_network)
    network_hash = compute_network_hash(current_network)

    # 2. Connexion à la base de données
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 3. Sauvegarde l'agent (si nouveau)
    cursor.execute("""
        INSERT OR IGNORE INTO agents (id_agent, id_parent, iteration_creation)
        VALUES (?, ?, ?)
    """, (agent_id, parent_agent_id, iteration))

    # 4. Vérifie si l'entrée existe déjà (pour éviter les doublons)
    cursor.execute("""
        SELECT 1 FROM networks
        WHERE id_agent = ? AND iteration = ?
    """, (agent_id, iteration))
    if cursor.fetchone() is None:  # Si pas de doublon, on insère
        # 5. Sauvegarde le réseau
        cursor.execute("""
            INSERT INTO networks (id_agent, iteration, network_data, network_hash, is_snapshot)
            VALUES (?, ?, ?, ?, ?)
        """, (agent_id, iteration, json.dumps(current_network), network_hash, is_snapshot))

        # 6. Sauvegarde le delta si parent existe
        if parent_agent_id:
            cursor.execute("""
                SELECT network_data
                FROM networks
                WHERE id_agent = ? AND iteration <= ?
                ORDER BY iteration DESC
                LIMIT 1
            """, (parent_agent_id, iteration))
            parent_network_data = cursor.fetchone()
            if parent_network_data:
                parent_network = json.loads(parent_network_data[0])
                diff = list(make_patch(parent_network, current_network))
                parent_commit_id = get_last_commit_id(cursor, agent_id)
                cursor.execute("""
                    INSERT INTO commits (id_agent, id_parent_commit, iteration, diff_data, message)
                    VALUES (?, ?, ?, ?, ?)
                """, (agent_id, parent_commit_id, iteration, json.dumps(diff), message))

    # 7. Valide les modifications
    conn.commit()
    conn.close()