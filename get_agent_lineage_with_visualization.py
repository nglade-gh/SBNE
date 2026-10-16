import sqlite3
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

NODE_LABELS = {
    0: "Œil (E)",
    1: "Bouche (M)",
    2: "Nageoire F",
    3: "Nageoire R",
    4: "R1",
    5: "R2",
    6: "R3",
    7: "R4",
}

NODE_ORDER = list(range(22))


def get_network_from_db(db_path, agent_id, iteration=999999):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT network_data, iteration
                   FROM networks
                   WHERE id_agent = ?
                   ORDER BY iteration DESC LIMIT 1
                   """, (int(agent_id),))
    result = cursor.fetchone()
    conn.close()
    if result:
        return json.loads(result[0]), result[1]
    return None, None


def get_ascendants(db_path, agent_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
                   WITH RECURSIVE ascendants AS (SELECT id_agent, id_parent, 1 as depth
                                                 FROM agents
                                                 WHERE id_agent = ?
                                                 UNION ALL
                                                 SELECT a.id_agent, a.id_parent, p.depth + 1
                                                 FROM agents a
                                                          JOIN ascendants p ON a.id_agent = p.id_parent
                                                 WHERE p.id_parent IS NOT NULL
                                                   AND a.id_agent
                       != p.id_agent
                       AND p.depth
                      < 100
                       )
                   SELECT DISTINCT id_agent
                   FROM ascendants
                   WHERE id_agent IS NOT NULL
                     AND id_agent != ?;
                   """, (int(agent_id), int(agent_id)))
    ascendants = [int(row[0]) for row in cursor.fetchall() if row[0] is not None]
    conn.close()
    return ascendants


def get_descendance(db_path, agent_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
                   WITH RECURSIVE descendants AS (SELECT id_agent, 1 as depth
                                                  FROM agents
                                                  WHERE id_parent = ?
                                                  UNION ALL
                                                  SELECT a.id_agent, d.depth + 1
                                                  FROM agents a
                                                           JOIN descendants d ON a.id_parent = d.id_agent
                                                  WHERE d.depth < 100)
                   SELECT DISTINCT id_agent
                   FROM descendants;
                   """, (int(agent_id),))
    descendants = [int(row[0]) for row in cursor.fetchall() if row[0] is not None]
    conn.close()
    return descendants


def get_fixed_positions(nodes):
    n = len(NODE_ORDER)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {NODE_ORDER[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}
    return {node: pos[node] for node in nodes if node in pos}


def visualize_network(network_data, agent_id, iteration, output_dir=None, is_parent=False, is_descendant=False):
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

    node_colors = ["#4CAF50" if states[n] > 0 else "#9E9E9E" for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=8, font_weight='bold')

    for (i, j), weight in zip(G.edges(), [weights[i][j] for i, j in G.edges()]):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(i, j)],
            edge_color="#4CAF50" if weight > 0 else "#F44336",
            width=abs(weight) * 2,
            arrowsize=20,
            connectionstyle='arc3,rad=0.1'
        )

        if i == j:
            x, y = pos[i]
            plt.text(x, y + 0.15, f"{weight:.1f}", fontsize=6, ha='center', va='center',
                     color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        else:
            x, y = (pos[i][0] + pos[j][0]) / 2, (pos[i][1] + pos[j][1]) / 2
            plt.text(x, y, f"{weight:.1f}", fontsize=6, ha='center', va='center',
                     color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    legend_elements = [
        Patch(facecolor='#4CAF50', edgecolor='#4CAF50', label='Nœud actif (état > 0)'),
        Patch(facecolor='#9E9E9E', edgecolor='#9E9E9E', label='Nœud inactif (état ≤ 0)'),
        Patch(facecolor='#4CAF50', edgecolor='#4CAF50', label='Poids > 0'),
        Patch(facecolor='#F44336', edgecolor='#F44336', label='Poids < 0'),
    ]

    title = f"Réseau neuronal - Agent {agent_id} (Itération {iteration})"
    if is_parent:
        title += " [PARENT]"
    elif is_descendant:
        title += " [DESCENDANT]"

    plt.title(title)
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.axis('off')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        prefix = "agent_"
        if is_parent:
            prefix = "ascendant_"
        elif is_descendant:
            prefix = "descendant_"
        output_path = os.path.join(output_dir, f"{prefix}{agent_id}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"💾 Image sauvegardée : {output_path}")

    plt.close()


def get_agent_lineage_with_visualization(db_path, agent_id, max_descendants=5):
    # CONVERSION FORCÉE EN INT dès l'entrée
    agent_id = int(agent_id)

    ascendants = get_ascendants(db_path, agent_id)
    descendants = get_descendance(db_path, agent_id)

    # Tout nettoyer en int, pas de None
    all_agents = [agent_id] + ascendants + descendants
    all_agents = [int(a) for a in all_agents if a is not None]

    unique_agents = set(all_agents)
    if len(all_agents) != len(unique_agents):
        print(f"⚠️ Attention : doublons détectés dans la lignée de l'agent {agent_id} !")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id_agent FROM agents")
    existing_agents = {int(row[0]) for row in cursor.fetchall() if row[0] is not None}
    conn.close()

    missing_agents = [a for a in unique_agents if a not in existing_agents]
    if missing_agents:
        print(f"⚠️ Attention : agents introuvables dans la base : {missing_agents}")

    output_dir = os.path.join(os.path.dirname(db_path), "lineage_visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # Agent principal
    network, iteration = get_network_from_db(db_path, agent_id)
    if network:
        visualize_network(network, agent_id, iteration, output_dir=output_dir)
        print(f"✅ Visualisé : agent {agent_id} (itération {iteration})")
    else:
        print(f"❌ Aucun réseau trouvé pour l'agent {agent_id}.")

    # Ascendants
    for ascendant_id in ascendants[::-1]:
        network, iteration = get_network_from_db(db_path, ascendant_id)
        if network:
            visualize_network(network, ascendant_id, iteration, output_dir=output_dir, is_parent=True)
            print(f"✅ Visualisé : ascendant {ascendant_id} (itération {iteration})")
        else:
            print(f"❌ Aucun réseau trouvé pour l'ascendant {ascendant_id}.")

    # Descendants
    for descendant_id in descendants[:max_descendants]:
        network, iteration = get_network_from_db(db_path, descendant_id)
        if network:
            visualize_network(network, descendant_id, iteration, output_dir=output_dir, is_descendant=True)
            print(f"✅ Visualisé : descendant {descendant_id} (itération {iteration})")
        else:
            print(f"❌ Aucun réseau trouvé pour le descendant {descendant_id}.")

    return {
        "agent_id": agent_id,
        "ascendants": ascendants[::-1],
        "descendants": descendants,
        "all_agents": all_agents,
        "unique_agents": list(unique_agents),
        "missing_agents": missing_agents
    }


if __name__ == "__main__":
    DB_PATH = "simulations/sim_20260709_120000/evolution.db"
    agent_id = 1120
    print(f"🔍 Test pour l'agent {agent_id}...")
    lineage = get_agent_lineage_with_visualization(DB_PATH, agent_id, max_descendants=5)

    print(f"\n🔍 Lignée de l'agent {agent_id} :")
    print(f"   - Ascendants : {lineage['ascendants']}")
    print(f"   - Descendants : {lineage['descendants'][:10]}... (Total: {len(lineage['descendants'])} agents)")
    print(f"   - Agents uniques : {len(lineage['unique_agents'])} (vs {len(lineage['all_agents'])} total)")
    print(f"   - Agents manquants : {lineage['missing_agents']}")