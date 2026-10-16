import sqlite3
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# =============================================
# CONFIGURATION
# =============================================
DB_PATH = "evolution.db"
OUTPUT_DIR = "lineage_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Labels des nœuds
NODE_LABELS = {
    0: "Œil (E)",
    1: "Bouche (M)",
    2: "Nageoire F",
    3: "Nageoire R",
    4: "R1",
    5: "R2",
    6: "R3",
    7: "R4",
    # Ajoute ici les labels pour R5, R6, etc. si nécessaire
}

# Ordre fixe des nœuds pour le layout circulaire (jusqu'à 22 nœuds)
NODE_ORDER = list(range(22))  # [0, 1, 2, ..., 21]

# =============================================
# 1. FONCTIONS POUR RÉCUPÉRER LES DONNÉES
# =============================================

def get_network_from_db(db_path, agent_id, iteration=999999):
    """Récupère le réseau d'un agent depuis la base de données."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT network_data, iteration
        FROM networks
        WHERE id_agent = ?
        ORDER BY iteration DESC
        LIMIT 1
    """, (agent_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return json.loads(result[0]), result[1]
    return None, None

def get_ascendants(db_path, agent_id):
    """Récupère TOUS les ascendants d'un agent."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        WITH RECURSIVE ascendants AS (
            SELECT id_agent, id_parent FROM agents WHERE id_agent = ?
            UNION ALL
            SELECT a.id_agent, a.id_parent FROM agents a
            JOIN ascendants p ON a.id_agent = p.id_parent
        )
        SELECT id_agent FROM ascendants
        WHERE id_agent IS NOT NULL AND id_agent != ?;
    """, (agent_id, agent_id))
    ascendants = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ascendants

def get_descendance(db_path, agent_id):
    """Récupère TOUS les descendants d'un agent."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        WITH RECURSIVE descendants AS (
            SELECT id_agent FROM agents WHERE id_parent = ?
            UNION ALL
            SELECT a.id_agent FROM agents a
            JOIN descendants d ON a.id_parent = d.id_agent
        )
        SELECT id_agent FROM descendants;
    """, (agent_id,))
    descendants = [row[0] for row in cursor.fetchall()]
    conn.close()
    return descendants

# =============================================
# 2. FONCTION POUR VISUALISER UN RÉSEAU
# =============================================

def get_fixed_positions(nodes):
    """Retourne les positions fixes pour les nœuds."""
    n = len(NODE_ORDER)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {NODE_ORDER[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}
    return {node: pos[node] for node in nodes if node in pos}

def visualize_network(network_data, agent_id, iteration, output_file=None, is_parent=False, is_descendant=False):
    """Visualise le réseau neuronal avec positions fixes, valeurs des poids et labels lisibles."""
    if not network_data:
        print(f"❌ Aucune donnée de réseau pour l'agent {agent_id}.")
        return

    states = network_data["states"]
    weights = network_data["weights"]
    nodes = list(range(len(states)))

    G = nx.DiGraph()
    for i in nodes:
        G.add_node(i, label=NODE_LABELS.get(i, f"N{i}"), state=states[i])

    for i in nodes:
        for j in nodes:
            if weights[i][j] != 0:
                G.add_edge(i, j, weight=weights[i][j])

    plt.figure(figsize=(12, 10))
    pos = get_fixed_positions(nodes)

    # Couleurs des nœuds
    node_colors = ["#4CAF50" if states[n] > 0 else "#9E9E9E" for n in G.nodes()]

    # Dessiner les nœuds et leurs labels
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=8, font_weight='bold')

    # Dessiner les arêtes et les poids
    for (i, j), weight in zip(G.edges(), [weights[i][j] for i, j in G.edges()]):
        # Dessiner l'arête
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(i, j)],
            edge_color="#4CAF50" if weight > 0 else "#F44336",
            width=abs(weight) * 2,
            arrowsize=20,
            connectionstyle='arc3,rad=0.1'
        )

        # Calculer la position du texte du poids
        if i == j:  # Auto-loop
            # Pour les auto-loops, placer le texte au-dessus du nœud
            x, y = pos[i]
            plt.text(x, y + 0.15, f"{weight:.1f}", fontsize=6, ha='center', va='center',
                     color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        else:
            # Pour les autres arêtes, placer le texte au milieu
            x, y = (pos[i][0] + pos[j][0]) / 2, (pos[i][1] + pos[j][1]) / 2
            plt.text(x, y, f"{weight:.1f}", fontsize=6, ha='center', va='center',
                     color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Ajout de la légende
    legend_elements = [
        Patch(facecolor='#4CAF50', edgecolor='#4CAF50', label='Nœud actif (état > 0)'),
        Patch(facecolor='#9E9E9E', edgecolor='#9E9E9E', label='Nœud inactif (état ≤ 0)'),
        Patch(facecolor='#4CAF50', edgecolor='#4CAF50', label='Poids > 0'),
        Patch(facecolor='#F44336', edgecolor='#F44336', label='Poids < 0'),
    ]

    # Indication parent/descendant dans le titre
    title = f"Réseau neuronal - Agent {agent_id} (Itération {iteration})"
    if is_parent:
        title += " [PARENT]"
    elif is_descendant:
        title += " [DESCENDANT]"

    plt.title(title)
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.axis('off')

    if output_file:
        plt.savefig(os.path.join(OUTPUT_DIR, output_file), dpi=200, bbox_inches='tight')
    plt.close()

# =============================================
# 3. FONCTION PRINCIPALE
# =============================================

def get_agent_lineage_with_visualization(db_path, agent_id, max_descendants=5):
    """Récupère les ascendants, descendants, visualise leurs réseaux et vérifie les données."""
    ascendants = get_ascendants(db_path, agent_id)
    descendants = get_descendance(db_path, agent_id)

    # Vérification : pas de doublons, pas d'agent inventé
    all_agents = [agent_id] + ascendants + descendants
    unique_agents = set(all_agents)
    if len(all_agents) != len(unique_agents):
        print(f"⚠️ Attention : doublons détectés dans la lignée de l'agent {agent_id} !")

    # Vérification : tous les agents existent dans la base de données
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id_agent FROM agents")
    existing_agents = {row[0] for row in cursor.fetchall()}
    conn.close()

    missing_agents = [a for a in all_agents if a not in existing_agents]
    if missing_agents:
        print(f"⚠️ Attention : agents introuvables dans la base : {missing_agents}")

    # Visualiser le réseau de l'agent principal
    network, iteration = get_network_from_db(db_path, agent_id)
    if network:
        visualize_network(network, agent_id, iteration, f"agent_{agent_id}.png")
        print(f"✅ Visualisé : agent {agent_id} (itération {iteration})")

    # Visualiser les réseaux des ascendants
    for ascendant_id in ascendants[::-1]:
        network, iteration = get_network_from_db(db_path, ascendant_id)
        if network:
            visualize_network(network, ascendant_id, iteration, f"ascendant_{ascendant_id}.png", is_parent=True)
            print(f"✅ Visualisé : ascendant {ascendant_id} (itération {iteration})")

    # Visualiser les réseaux des descendants
    for descendant_id in descendants[:max_descendants]:
        network, iteration = get_network_from_db(db_path, descendant_id)
        if network:
            visualize_network(network, descendant_id, iteration, f"descendant_{descendant_id}.png", is_descendant=True)
            print(f"✅ Visualisé : descendant {descendant_id} (itération {iteration})")

    return {
        "agent_id": agent_id,
        "ascendants": ascendants[::-1],
        "descendants": descendants,
        "all_agents": all_agents,
        "unique_agents": list(unique_agents),
        "missing_agents": missing_agents
    }

# =============================================
# 4. EXEMPLE D'UTILISATION
# =============================================
if __name__ == "__main__":
    agent_id = 15000  # Remplace par l'ID de ton agent
    print(f"🔍 Test pour l'agent {agent_id}...")
    DB_PATH = os.path.join("simulations", "sim_20260710_100905", "evolution.db")
    lineage = get_agent_lineage_with_visualization(DB_PATH, agent_id, max_descendants=5)

    print(f"\n🔍 Lignée de l'agent {agent_id} :")
    print(f"   - Ascendants : {lineage['ascendants']}")
    print(f"   - Descendants : {lineage['descendants'][:10]}... (Total: {len(lineage['descendants'])} agents)")
    print(f"   - Agents uniques : {len(lineage['unique_agents'])} (vs {len(lineage['all_agents'])} total)")
    print(f"   - Agents manquants : {lineage['missing_agents']}")

    print(f"\n📁 Les images ont été enregistrées dans le dossier '{OUTPUT_DIR}'.")