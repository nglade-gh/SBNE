import math

class DiffusionGrid:
    """
    Grille servant uniquement à décider COMMENT répartir spatialement
    une quantité d'énergie déjà existante (nourriture initiale ou déchets
    de digestion) entre une case et ses voisines orthogonales.

    Elle ne stocke aucun état persistant et ne crée / ne détruit jamais
    d'énergie : elle calcule juste des "dépôts" (position, quantité)
    dont la somme est TOUJOURS égale à l'énergie reçue en entrée.
    """

    def __init__(self, width_px, height_px, cell_size, diffusion_rate=2/3, seuil_diffusion=200):
        self.width_px = width_px
        self.height_px = height_px
        self.cell_size = cell_size
        # Nombre de cases de la grille (arrondi au supérieur pour couvrir tout l'environnement)
        self.cols = max(1, math.ceil(width_px / cell_size))
        self.rows = max(1, math.ceil(height_px / cell_size))
        # Fraction de l'énergie totale répartie vers les voisins (le reste, 1 - diffusion_rate, reste au centre)
        self.diffusion_rate = diffusion_rate
        # En dessous (ou à) ce seuil, l'énergie n'est plus fragmentée : elle reste entière sur une seule case
        # (évite l'accumulation de miettes trop petites pour être efficacement consommées)
        self.seuil_diffusion = seuil_diffusion

    def _cell_coords(self, x, y):
        """Convertit une position en pixels en indices de case, bornés à la grille."""
        cx = int(x // self.cell_size)
        cy = int(y // self.cell_size)
        cx = min(max(cx, 0), self.cols - 1)
        cy = min(max(cy, 0), self.rows - 1)
        return cx, cy

    def _cell_center(self, cx, cy):
        """Retourne les coordonnées pixel du centre d'une case."""
        return (cx * self.cell_size + self.cell_size / 2,
                cy * self.cell_size + self.cell_size / 2)

    def compute_deposits(self, x, y, energy):
        cx, cy = self._cell_coords(x, y)

        if energy <= self.seuil_diffusion:
            center_x, center_y = self._cell_center(cx, cy)
            # Correction pour le tore
            center_x = center_x % self.width_px
            center_y = center_y % self.height_px
            return [((center_x, center_y), energy)]

        voisins = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < self.cols and 0 <= ny < self.rows:
                voisins.append((nx, ny))

        nb_voisins = len(voisins)

        if nb_voisins == 0:
            center_x, center_y = self._cell_center(cx, cy)
            center_x = center_x % self.width_px
            center_y = center_y % self.height_px
            return [((center_x, center_y), energy)]

        energie_centre = energy * (1 - self.diffusion_rate)
        energie_par_voisin = (energy * self.diffusion_rate) / nb_voisins

        deposits = []
        center_x, center_y = self._cell_center(cx, cy)
        center_x = center_x % self.width_px
        center_y = center_y % self.height_px
        deposits.append(((center_x, center_y), energie_centre))

        for (nx, ny) in voisins:
            neighbor_x, neighbor_y = self._cell_center(nx, ny)
            neighbor_x = neighbor_x % self.width_px
            neighbor_y = neighbor_y % self.height_px
            deposits.append(((neighbor_x, neighbor_y), energie_par_voisin))

        return deposits