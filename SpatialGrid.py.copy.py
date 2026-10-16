# On applique ces méthodes, car boucler sur tous les agents alors que certain sont très loin nous consommaient trop de ressources (bcp trop !)
def update_grid(agents, foods, cell_size):
    grid = {}

    # On crée une liste globale contenant d'office les agents
    entites = agents.copy()

    # On ajoute la nourriture active à cette liste
    entites.extend([f for f in foods if f.active])

    for agent in entites:
        # Calcul de l'index de la case
        cx = int(agent.x // cell_size)
        cy = int(agent.y // cell_size)

        cell_key = (cx, cy)
        if cell_key not in grid:
            grid[cell_key] = []
        grid[cell_key].append(agent)
    return grid


def get_neighbors(agent, grid, cell_size):
    neighbors = []
    cx = int(agent.x // cell_size)
    cy = int(agent.y // cell_size)

    # On boucle sur les 9 cases (celle de l'agent + les 8 voisines)
    for i in range(cx - 1, cx + 2):
        for j in range(cy - 1, cy + 2):
            if (i, j) in grid:
                neighbors.extend(grid[(i, j)])

    return neighbors
