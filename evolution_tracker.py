import json
import sqlite3
import hashlib
from jsonpatch import make_patch, apply_patch

# --- Gestion des connexions (une seule connexion SQLite par base, réutilisée) ---
_connections = {}


def _get_connection(db_path):
    conn = _connections.get(db_path)
    if conn is None:
        conn = sqlite3.connect(db_path)
        _connections[db_path] = conn
    return conn


def close_all_connections():
    """Ferme toutes les connexions SQLite ouvertes par ce module."""
    for conn in _connections.values():
        conn.close()
    _connections.clear()


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
        SELECT id
        FROM commits
        WHERE id_agent = ?
        ORDER BY iteration DESC, id DESC
        LIMIT 1
    """, (agent_id,))
    result = cursor.fetchone()
    return result[0] if result else None

# --- Sauvegarde ---
def save_network_state(db_path, agent_id, parent_agent_id, sb_network, iteration, is_snapshot=False, message="Update"):
    current_network = extract_network(sb_network)
    network_hash = compute_network_hash(current_network)

    conn = _get_connection(db_path)
    cursor = conn.cursor()

    # 1. Sauvegarder l'agent (si nouveau)
    cursor.execute("""
        INSERT OR IGNORE INTO agents (id_agent, id_parent, iteration_creation)
        VALUES (?, ?, ?)
    """, (agent_id, parent_agent_id, iteration))

    #  VÉRIFIER SI L'ENTRÉE EXISTE DÉJÀ (pour éviter les doublons)
    cursor.execute("""
        SELECT 1 FROM networks
        WHERE id_agent = ? AND iteration = ?
    """, (agent_id, iteration))
    if cursor.fetchone() is None:  # Si pas de doublon, on insère
        # 2. Sauvegarder le réseau
        cursor.execute("""
            INSERT INTO networks (id_agent, iteration, network_data, network_hash, is_snapshot)
            VALUES (?, ?, ?, ?, ?)
        """, (agent_id, iteration, json.dumps(current_network), network_hash, is_snapshot))

        # 3. Sauvegarder le delta si parent existe
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

    conn.commit()
    # La connexion est volontairement laissée ouverte (réutilisation par le cache) :
    # elle sera fermée par close_all_connections().
# --- Reconstruction ---
def rebuild_network(db_path, agent_id, target_iteration):
    """
    Reconstruire le réseau d'un agent à une itération donnée.
    Args:
        db_path: Chemin vers la base de données SQLite.
        agent_id: ID de l'agent.
        target_iteration: Itération cible.
    Returns:
        Dictionnaire avec "states" et "weights".
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()

    # 1. Récupérer le dernier snapshot avant ou à target_iteration
    cursor.execute("""
        SELECT network_data, iteration
        FROM networks
        WHERE id_agent = ? AND iteration <= ? AND is_snapshot = TRUE
        ORDER BY iteration DESC
        LIMIT 1
    """, (agent_id, target_iteration))
    snapshot_data = cursor.fetchone()

    if not snapshot_data:
        # Pas de snapshot : prendre le premier état
        cursor.execute("""
            SELECT network_data, iteration
            FROM networks
            WHERE id_agent = ? AND iteration <= ?
            ORDER BY iteration ASC
            LIMIT 1
        """, (agent_id, target_iteration))
        snapshot_data = cursor.fetchone()

    if not snapshot_data:
        raise ValueError(f"Aucun réseau trouvé pour l'agent {agent_id} avant l'itération {target_iteration}")

    current_network = json.loads(snapshot_data[0])
    snapshot_iteration = snapshot_data[1]

    # 2. Récupérer tous les deltas entre snapshot_iteration et target_iteration
    cursor.execute("""
        SELECT diff_data
        FROM commits
        WHERE id_agent = ? AND iteration > ? AND iteration <= ?
        ORDER BY iteration
    """, (agent_id, snapshot_iteration, target_iteration))
    diffs = [json.loads(row[0]) for row in cursor.fetchall()]

    # 3. Appliquer les deltas séquentiellement (toutes les opérations jsonpatch :
    #    replace, add, remove — les changements de taille du réseau compris)
    for diff in diffs:
        if diff:
            current_network = apply_patch(current_network, diff)

    return current_network

# --- Analyse ---
def get_network_clusters(db_path):
    """Récupérer tous les clusters de réseaux identiques."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM network_clusters WHERE agent_count > 1")
    clusters = cursor.fetchall()
    conn.close()
    return clusters

def get_agents_with_unchanged_network(db_path):
    """Récupérer les agents dont le réseau est identique à celui de leur parent."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.id_agent, a.id_parent, n.network_hash, n.iteration
        FROM agents a
        JOIN networks n ON a.id_agent = n.id_agent
        JOIN networks p ON a.id_parent = p.id_agent
        WHERE n.network_hash = p.network_hash
        AND n.iteration > p.iteration
    """)
    results = cursor.fetchall()
    conn.close()
    return results