import numpy as np
import random

class SbNetwork:
    def __init__(self):
        self.num_nodes=8
        # Création de la matrice des états
        self.states = np.zeros(self.num_nodes, dtype=int)
        # Signification des positions dans le tableau
        # 0. Œil
        # 1. Bouche
        # 2. Nageoire F (Forward)
        # 3. Nageoire R (Rotate)
        # 4. R1
        # 5. R2
        # 6. R3
        # 7. R4
        
        # Création de la matrice des poids
        self.weights = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        
        # Initialisation des états
        self.states[2] = 1 # Nageoire F est tout le temps activé
        self.states[3] = 1 # Nageoire R est activé à l'initialisation
        
        # Initialisation des poids
        self.weights[0, 1] = 1 # Oeil active la bouche
        self.weights[2, 2] = 1 # Nageoire F active Nageoire F
        self.weights[3, 3] = -1 # R s'auto inhibe
        self.weights[3, 4] = 1 # R active R1...
        self.weights[4, 4] = -1
        self.weights[4, 5] = 1
        self.weights[5, 5] = -1
        self.weights[5, 6] = 1
        self.weights[6, 6] = -1
        self.weights[6, 7] = 1
        self.weights[7, 7] = -1
        self.weights[7, 3] = 1
        
        # On stocke les id réelles de chaque nœud (pour la visualisation).
        self.true_ids = list(range(self.num_nodes)) # Donne [0, 1, 2, 3, 4, 5, 6, 7]
        self.next_historical_id = 8 # Le prochain neurone créé s'appellera N8
        
    def step(self, oeil):
        # On récupère bien la valeur de l'œil par rapport à l'environnement
        self.states[0] = oeil
        
        # Calcul de la somme pondérée pour tous les nœuds d'un coup (Produit matriciel)
        x = self.states @ self.weights
        
        # On applique un seuil (si x > 0, on met 1, sinon 0).
        new_states = (x > 0).astype(int)
        
        # On remet la bonne valeur de l'œil
        new_states[0] = oeil
        
        # Mise à jour synchrone
        self.states = new_states
        
        # On return que les états actionneurs
        return self.states[1], self.states[2], self.states[3]

    def mutation(self, pd, pi, wmax):
        if random.random() < pd:
            self.deletion()
        if random.random() < pi:
            self.insertion(wmax)
    
    def evolution(self, pw, wmax):
        if random.random() < pw:
            self.weight_update(wmax)
    
    def weight_update(self, wmax):
        entry_node = random.randint(0, self.num_nodes-1)
        exit_node = random.randint(0, self.num_nodes-1)
        self.weights[entry_node, exit_node] += random.randint(-1, 1)
        #self.weights[entry_node, exit_node] = np.clip(self.weights[entry_node, exit_node], -wmax, wmax)
            
    def insertion(self, wmax):
        n = self.num_nodes
        # On ajoute un nœud au réseau de neurones
        self.states = np.append(self.states, random.randint(0, 1))
        # On crée une nouvelle matrice de poids d'un cran plus grande
        new_weights = np.zeros((n+1, n+1), dtype=int)
        # On copie l'ancienne matrice dans la nouvelle
        new_weights[:n, :n] = self.weights
        entry_node = random.randint(0, self.num_nodes-1)
        exit_node = random.randint(0, self.num_nodes-1)
        new_weights[entry_node, n] = random.randint(-wmax, wmax)
        new_weights[n, exit_node] = random.randint(-wmax, wmax)
        # On remplace l'ancienne matrice de poids par la nouvelle
        self.weights = new_weights
        self.num_nodes += 1 # On met à jour le nombre de nœuds
        self.true_ids.append(self.next_historical_id) # On ajoute le nouveau nœud dans les id
        self.next_historical_id += 1 # On met à jour l'id du prochain nœud
        
    def deletion(self):
        if self.num_nodes <=4:
            return
        
        node_to_delete = random.randint(4, self.num_nodes-1)
        self.states = np.delete(self.states, node_to_delete)
        self.weights = np.delete(self.weights, node_to_delete, axis=0)
        self.weights = np.delete(self.weights, node_to_delete, axis=1)
        self.num_nodes -=1
        self.true_ids.pop(node_to_delete)