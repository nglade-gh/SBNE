import numpy as np
import csv
import os

def save_nodes_influence(pop, iteration, test_name):
    folder = os.path.join(test_name, "exports")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    header = [
        "ID_Agent", "Pos_X", "Pos_Y",
        "Oeil_Avancer", "Oeil_Rotation",
        "Avancer_Rotation", "Rotation_Avancer",
        "Avancer_Oeil", "Rotation_Oeil"
    ]
    
    file_name = os.path.join(folder, f"agents_save_{iteration}.csv")
    
    with open(file_name, mode="w", newline='', encoding="utf-8") as fichier_csv:
        writer = csv.writer(fichier_csv, delimiter=',')
        writer.writerow(header)
        
        max_depth = 5 
        
        for agent in pop:
            w = agent.sbn.weights
        
            # On initialise la matrice d'influence en float
            mi = np.zeros_like(w, dtype=float)
            
            # Matrice qui va stocker les puissances successives
            w_power = np.eye(agent.sbn.num_nodes)
            
            # On utilise la profondeur fixe au lieu de num_nodes
            for k in range(1, max_depth + 1):
                w_power = np.matmul(w_power, w)
                mi += w_power
            
            # Récupération des données spatiales et d'identification
            id_agent = agent.id
            pos_x = agent.x
            pos_y = agent.y
            
            # Extraction avec tes vrais index :
            # 0=Œil, 1=Bouche, 2=Nageoire F (Avancer), 3=Nageoire R (Rotation)
            o_a = mi[0, 2]  # Oeil -> Avancer
            o_r = mi[0, 3]  # Oeil -> Rotation
            a_r = mi[2, 3]  # Avancer -> Rotation
            r_a = mi[3, 2]  # Rotation -> Avancer
            a_o = mi[2, 0]  # Avancer -> Oeil
            r_o = mi[3, 0]  # Rotation -> Oeil
            
            # Création et écriture de la ligne
            ligne_agent = [
                id_agent, pos_x, pos_y,
                o_a, o_r, a_r, r_a, a_o, r_o
            ]
            writer.writerow(ligne_agent)
            
    print(f"Sauvegarde des cerveaux générée à l'itération : {iteration} dans le fichier {file_name}")