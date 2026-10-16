import math


class SpatialGrid:
    """
    Grille spatiale (environnement torique) pour la recherche de voisins.

    Plutôt que de parcourir tous les agents à chaque pas (coût O(n²)), les
    entités sont rangées par case ; une requête ne regarde que les 9 cases
    autour de l'entité. L'environnement étant un tore, les cases s'enroulent
    (un agent près du bord voit les entités de l'autre côté).
    """

    def __init__(self, width_px, height_px, cell_size):
        self.width_px = width_px
        self.height_px = height_px
        self.cell_size = cell_size
        # Nombre de cases de la grille (arrondi au supérieur pour couvrir tout l'environnement)
        self.cols = max(1, math.ceil(width_px / cell_size))
        self.rows = max(1, math.ceil(height_px / cell_size))
        self.grid = {}

    def _cell_key(self, x, y):
        """Index de case d'une position, enroulé sur le tore."""
        return (int(x // self.cell_size) % self.cols,
                int(y // self.cell_size) % self.rows)

    def rebuild(self, agents, foods):
        """Reconstruit la grille depuis les agents et la nourriture active et vivante."""
        self.grid = {}
        for entity in agents:
            self._add(entity)
        for food in foods:
            if food.active and food.alive:
                self._add(food)

    def _add(self, entity):
        key = self._cell_key(entity.x, entity.y)
        self.grid.setdefault(key, []).append(entity)

    def neighbors(self, entity):
        """Entités situées dans la case de l'entité + les 8 cases voisines (enroulées)."""
        cx, cy = self._cell_key(entity.x, entity.y)
        result = []
        for i in (cx - 1, cx, cx + 1):
            for j in (cy - 1, cy, cy + 1):
                cell = self.grid.get((i % self.cols, j % self.rows))
                if cell:
                    result.extend(cell)
        return result
