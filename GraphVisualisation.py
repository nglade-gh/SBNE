import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Fonction permettant d'afficher le graphique du réseau de neurones d'un agent
def show_sbn_graph(agent_id, sbn):
    plt.figure(figsize=(18, 8))
    plt.suptitle(f"Cerveau de l'agent {agent_id} (Noeuds: {sbn.num_nodes})", fontsize=16)
    
    # Création du graphe orienté
    g = nx.DiGraph()
    
    # Dictionnaire de traduction
    name_nodes = {0: "E", 1: "M", 2: "F", 3: "R", 4: "R1", 5: "R2", 6: "R3", 7: "R4"}
    
    # On ajoute des noms génériques (N8, N9...) si le réseau a muté et grandi
    for i in range(8, sbn.num_nodes):
        name_nodes[i] = f"N{i}"

    # On crée chaque nœud de haut en bas en fonction de la fonction (pour plus de lisibilité).
    for ida in sbn.true_ids:
        name = name_nodes.get(ida, f"N{ida}")
        if ida in [0, 1]:         # Oeil, Bouche
            layer = 3           # Couche supérieure
        elif ida in [4, 5, 6]:    # R1, R2, R3
            layer = 1           # Couche inférieure
        elif ida in [3, 7]:       # R, R4
            layer = 0           # Couche inférieure
        else:                   # F et les nouveaux nœuds (N8, N9...)
            layer = 2           # Couche intermédiaire
            
        g.add_node(name, layer=layer)

    # On parcourt de la matrice de poids
    edges_green = []
    edges_red = []
    
    for i in range(sbn.num_nodes):
        for j in range(sbn.num_nodes):
            poids = sbn.weights[i, j]
            if poids != 0:
                id_depart = sbn.true_ids[i]
                id_arrivee = sbn.true_ids[j]
                
                # On génère le nom dynamiquement s'il est nouveau (>= 8).
                nom_depart = name_nodes.get(id_depart, f"N{id_depart}")
                nom_arrivee = name_nodes.get(id_arrivee, f"N{id_arrivee}")
                
                # On ajoute la connexion avec son poids
                g.add_edge(nom_depart, nom_arrivee, weight=poids)
                
                # On trie pour l'affichage des couleurs
                if poids > 0:
                    edges_green.append((nom_depart,nom_arrivee))
                else:
                    edges_red.append((nom_depart, nom_arrivee))

    # align="horizontal" force les couches à se placer sur des lignes horizontales
    pos = nx.multipartite_layout(g, subset_key="layer", align="horizontal")

    noeuds_bas = ["R", "R1", "R2", "R3", "R4"]
    # On récupère toutes les coordonnées X de la couche du bas et on les trie (pour ne pas avoir R, R2, R1, R4, R3 mais bien R,R1,...,R4).
    x_coords_bas = sorted([pos[n][0] for n in noeuds_bas if n in pos])
    
    # On réassigne ces coordonnées X triées à nos nœuds dans le bon ordre
    current_idx = 0
    for noeud in noeuds_bas:
        if noeud in pos:
            pos[noeud] = np.array([x_coords_bas[current_idx], pos[noeud][1]])
            current_idx += 1
    
    # 2. Aligner E, F et R verticalement sur la gauche
    # On prend la coordonnée X la plus à gauche (qui appartient maintenant à R)
    x_gauche = pos["R"][0]
    pos["E"] = np.array([x_gauche, pos["E"][1]])
    pos["F"] = np.array([x_gauche, pos["F"][1]])
    
    # Calcul dynamique des tailles de nœud, arrête et texte
    # Facteur d'échelle : 1.0 s'il y a 8 nœuds, 0.5 s'il y en a 16, etc.
    facteur_echelle = 8 / max(8, sbn.num_nodes)
    
    # On définit des limites minimales pour que ça reste visible à l'écran
    taille_noeud = max(500, int(2000 * facteur_echelle))
    taille_police = max(6, int(10 * facteur_echelle))
    taille_fleche = max(10, int(20 * facteur_echelle))
    
    # 1. Dessin des nœuds avec la taille dynamique
    nx.draw_networkx_nodes(g, pos, node_color='lightblue', node_size=taille_noeud)
    nx.draw_networkx_labels(g, pos, font_size=taille_police, font_weight="bold")
    
    # 2. Dessin des arêtes (flèches)
    nx.draw_networkx_edges(g, pos, edgelist=edges_green, edge_color='green',
                           arrows=True, arrowsize=taille_fleche, width=2,
                           node_size=taille_noeud)
                           
    nx.draw_networkx_edges(g, pos, edgelist=edges_red, edge_color='red',
                           arrows=True, arrowsize=taille_fleche, width=2,
                           node_size=taille_noeud)
    
    # On récupère les poids de toutes les arêtes
    edge_labels = nx.get_edge_attributes(g, 'weight')
    
    # On dessine toutes les étiquettes manuellement
    for (u, v), weight in edge_labels.items():
        texte_poids = f"{weight:.2f}"
        
        if u == v:
            # Boucle (Self-loop) : on décale le texte au-dessus du nœud
            x, y = pos[u]
            plt.text(x, y + 0.15, texte_poids, fontsize=8, color='black', 
                     ha='center', va='center', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        else:
            # Connexion normale : on calcule le centre géométrique parfait
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2
            
            # On décale le texte en fonction de l'ordre alphabétique (cela empêche les flèches "Aller" et les flèches "Retour" de superposer leurs étiquettes).
            if str(u) < str(v):
                x_mid += 0.02
                y_mid += 0.04
            else:
                x_mid -= 0.02
                y_mid -= 0.04
                
            plt.text(x_mid, y_mid, texte_poids, fontsize=8, color='black', 
                     ha='center', va='center', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    # Affichage de la fenêtre
    plt.show()

# Affichage des statistiques global de la simulation
def show_simulation_summary(history_steps, history_pop, history_size, history_energy, history_nodes_activated, history_global_energy, chemin_sauvegarde):
    if not history_steps:
        # Pas encore de données (simulation trop courte ou population éteinte avant le premier relevé)
        plt.figure(figsize=(12, 8))
        plt.suptitle("Bilan de la Simulation (aucune donnée)", fontsize=16)
        if chemin_sauvegarde:
            plt.savefig(chemin_sauvegarde, dpi=150)
        plt.close()
        return

    plt.figure(figsize=(12, 8))
    plt.suptitle("Bilan de la Simulation", fontsize=16)

    # --- GRAPHE 1 : Évolution de la Population ---
    plt.subplot(3, 2, 1)
    plt.plot(history_steps, history_pop, color='green')
    plt.title("Population totale")
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Nombre d'agents")
    plt.grid(True, alpha=0.3)

    # --- GRAPHE 2 : Taille moyenne du cerveau (SBN) ---
    plt.subplot(3, 2, 2)
    plt.plot(history_steps, history_size, color='blue')
    plt.title("Taille moyenne du réseau (Noeuds)")
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Nombre de neurones")
    plt.grid(True, alpha=0.3)

    # --- GRAPHE 3 : Énergie moyenne des agents ---
    plt.subplot(3, 2, 3)
    plt.plot(history_steps, history_energy, color='orange')
    plt.title("Énergie moyenne")
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Unités d'énergie")
    plt.grid(True, alpha=0.3)

    # --- GRAPHE 4 : Nombre de nœuds moyen activé par agents ---
    plt.subplot(3, 2, 4)
    plt.plot(history_steps, history_nodes_activated, color='red')
    plt.title("Nombre de neurones moyen activé")
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Nombre de neurones")
    plt.grid(True, alpha=0.3)
    
    # --- GRAPHE 5 : Énergie globale du système ---
    plt.subplot(3, 2, 5)
    plt.plot(history_steps, history_global_energy, color='purple')
    plt.title("Énergie global du système")
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Énergie disponible")
    # Fenêtre ±10 autour de l'énergie tant que la dérive est petite ;
    # sinon, fenêtre dynamique englobant toutes les valeurs.
    val_min = min(history_global_energy)
    val_max = max(history_global_energy)
    span = max(val_max - val_min, 20.0)
    centre = (val_min + val_max) / 2
    plt.ylim(centre - span / 2, centre + span / 2)
    plt.ticklabel_format(useOffset=False, style='plain', axis='y')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if chemin_sauvegarde:
        plt.savefig(chemin_sauvegarde, dpi=150)
    plt.close()