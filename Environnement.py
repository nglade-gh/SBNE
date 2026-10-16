"""
SBNE — Simulation d'un écosystème évolutionnaire.

Un environnement en tore (largeur × hauteur) contient un nombre variable
d'agents. Chaque agent possède une énergie qu'il consomme (métabolisme,
neurones activés, actions) et qu'il retrouve en mangeant (nourriture du sol
ou autres agents). Les actions (manger, avancer, tourner) sont pilotées par
un réseau de neurones booléen signé (SbNetwork) qui évolue à la reproduction
(division + mutation). L'historique des réseaux est enregistré dans une base
SQLite sous forme de snapshots + deltas (jsonpatch, façon git).

Lancement :
    python Environnement.py                  # avec fenêtre graphique
    python Environnement.py --headless       # sans fenêtre (calcul seul)
    python Environnement.py --max-steps 5000 # arrêt automatique
    python Environnement.py --seed 7         # surcharge de la graine
"""
import argparse
import json
import math
import os
import random
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pygame

from Agent import Agent
from DiffusionGrid import DiffusionGrid
from ExportData import save_nodes_influence
from Food import Food
from GraphVisualisation import show_simulation_summary
from Interface import draw_dashboard, graphics_pause, show_graphics_off
from SpatialGrid import SpatialGrid
from evolution_tracker import save_network_state, close_all_connections

# Dashboard d'information
DASHBOARD_SIZE = 50

# =============================================
# PARAMÈTRES DE LA SIMULATION
# =============================================
PARAMS = {
    "TEST_NAME": "No_weights_cap_higher_cost_neuron",
    "SEED": 42,
    "NUM_AGENTS": 150,
    "MAX_ITERATIONS": 0,  # Temps maximal : nombre max de pas de simulation (0 = illimité). Surchargeable via --max-steps
    "BASE_ENERGY": 1000,  # init 1000
    "DIVISION_ENERGY": 1200,  # j'abaisse de 3000 à 1500 d'énergie pour faciliter la diversité
    "PROBA_DELETION": 0.4,  # on passe de 0,05 à 0,1.
    "PROBA_INSERTION": 0.4,  # on passe de 0,05 à 0,1.
    "PROBA_EVOLUTION": 0.4,  # on passe de 0,05 à 0,1.
    "VALEUR_MAX_POIDS": 3,
    "DISTANCE_VISION": 70,
    "VISION_ANGLE": 30,
    "COST_NEURON": 0.3,
    "COST_MOVE": 0.02,
    "COST_ROTATE": 0.1,
    "COST_EAT": 0.01,
    "COST_METABOLISM": 0.1,
    "NUM_FOOD": 10,
    "DIGESTION_INTERVAL": 120,
    "DIGESTION_MIN": 250,
    "DIFFUSION_RATE": 2/3,  # Fraction de l'énergie déposée qui part vers les cases voisines (le reste reste au centre)
    "SEUIL_DIFFUSION": 250  # En dessous (ou à) ce seuil, l'énergie n'est plus fragmentée (évite les miettes trop petites) — doit rester > DIGESTION_MIN pour être efficace
}


def init_database(db_path):
    """Crée les tables (et index) de la base SQLite si elles n'existent pas."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            id_agent TEXT PRIMARY KEY,
            id_parent TEXT,
            iteration_creation INTEGER,
            FOREIGN KEY (id_parent) REFERENCES agents(id_agent)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS networks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_agent TEXT,
            iteration INTEGER,
            network_data TEXT,
            network_hash TEXT,
            is_snapshot BOOLEAN,
            FOREIGN KEY (id_agent) REFERENCES agents(id_agent)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS commits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_agent TEXT,
            id_parent_commit INTEGER,
            iteration INTEGER,
            diff_data TEXT,
            message TEXT,
            FOREIGN KEY (id_agent) REFERENCES agents(id_agent)
        )
    """)
    # Index pour les requêtes de reconstruction (rebuild_network, deltas)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_networks_agent_iter ON networks(id_agent, iteration)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_commits_agent_iter ON commits(id_agent, iteration)")
    conn.commit()
    conn.close()


class Simulation:
    """Contient tout l'état d'une simulation et orchestre logique / rendu / base de données."""

    def __init__(self, params, headless=False, max_steps=0, outdir=None):
        # --- PARAMÈTRES GÉNÉRAUX ---
        self.test_name = str(params.get("TEST_NAME", "Extinction_Initiale"))  # Nom de la session ou de l'expérience en cours
        seed = params.get("SEED", 42)  # Graine de génération aléatoire (assure que la simulation est reproductible)

        # --- MOTEUR ET AFFICHAGE ---
        self.width = int(params.get("WIDTH", 1280))  # Largeur de la fenêtre de simulation en pixels
        self.height = int(params.get("HEIGHT", 720))  # Hauteur de la fenêtre de simulation en pixels
        self.n_iter = int(params.get("N_ITER",
                                     1))  # Nombre de mises à jour logiques par image (accélère le temps simulé sans affecter l'affichage)
        self.fps = int(params.get("FPS", 60))  # Limite d'images par seconde pour le rendu visuel
        self.display_fps = params.get("DISPLAY_FPS", 0)  # Active (1) ou désactive (0) l'affichage du compteur visuel des FPS

        # --- PHYSIQUE ET CAPACITÉS DES AGENTS ---
        self.taille_agent = int(params.get("TAILLE_AGENT", 5))  # Rayon d'affichage du cercle représentant l'agent
        self.rotate_deg = params.get("ROTATE_DEG",
                                     10)  # Vitesse de rotation : nombre de degrés modifiés quand l'agent décide de tourner
        self.vision_angle = params.get("VISION_ANGLE", 45)  # Ouverture du cône de vision de l'agent (en degrés)
        self.distance_manger = int(
            params.get("DISTANCE_MANGER", self.taille_agent * 2))  # Portée maximale à laquelle l'agent peut attaquer/manger
        self.distance_vision = int(
            params.get("DISTANCE_VISION", self.distance_manger))  # Portée maximale à laquelle l'agent détecte les autres entités

        # --- DYNAMIQUE DE POPULATION ET ÉNERGIE ---
        self.num_agents = int(params.get("NUM_AGENTS", 1000))  # Taille de la population de départ générée aléatoirement
        self.base_energy = params.get("BASE_ENERGY", 100)  # Capital énergétique de départ pour les premiers agents
        self.division_energy = params.get("DIVISION_ENERGY",
                                          600)  # Palier d'énergie requis pour déclencher la reproduction par division
        self.num_food = int(params.get("NUM_FOOD", 100))
        self.cost_move = params.get("COST_MOVE", 1)
        self.cost_rotate = params.get("COST_ROTATE", 1)
        self.cost_eat = params.get("COST_EAT", 1)
        self.cost_neuron = params.get("COST_NEURON", 1)
        self.cost_metabolism = params.get("COST_METABOLISM", 0.05)
        self.digestion_min = params.get("DIGESTION_MIN", 40)
        self.digestion_rate = params.get("DIGESTION_RATE", 5)
        self.digestion_interval = params.get("DIGESTION_INTERVAL", 10)
        self.active_corpse = params.get("ACTIVE_CORPSE", True)
        self.diffusion_rate = params.get("DIFFUSION_RATE", 2/3)
        self.seuil_diffusion = params.get("SEUIL_DIFFUSION", 300)

        # --- MUTATIONS DU CERVEAU (RÉSEAU DE NEURONES) ---
        self.proba_deletion = params.get("PROBA_DELETION",
                                         0.01)  # Probabilité qu'un agent perde un nœud neuronal lors d'une mutation
        self.proba_insertion = params.get("PROBA_INSERTION", 0.02)  # Probabilité qu'un agent développe un nouveau nœud neuronal
        self.proba_evolution = params.get("PROBA_EVOLUTION", 0.02)  # Probabilité qu'un agent développe un nouveau nœud neuronal
        self.valeur_max_poids = params.get("VALEUR_MAX_POIDS", 3)  # Valeur absolue maximale des connexions créées entre les neurones

        # --- STATISTIQUE ET VISUALISATION ---
        self.tracking_id = params.get("TRACKING_ID", None)

        # Paramètre de la grille spatiale (recherche de voisins, indépendante de la diffusion de nourriture)
        self.cell_size = self.distance_vision * 1.2  # Doit être >= DISTANCE_VISION (on met distance_vision + 20% pour avoir de la marge d'erreur)

        # Grille de diffusion de nourriture : résolution fine, basée sur la taille d'un agent, indépendante de CELL_SIZE
        self.food_cell_size = self.taille_agent
        self.diffusion_grid = DiffusionGrid(self.width, self.height, self.food_cell_size,
                                            self.diffusion_rate, self.seuil_diffusion)  # diffusion nourriture

        random.seed(seed)  # On applique la seed à random (cela marchera peu importe ou random est appelé (n'importe quel classe))
        np.random.seed(seed)  # Pareil pour le random de numpy

        # --- CHEMINS : dossier de simulation + base de données ---
        # Nom de dossier explicite (lanceur de campagnes) ou horodatage par défaut
        folder_name = outdir if outdir else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.simulation_folder = os.path.join("simulations", folder_name)
        os.makedirs(self.simulation_folder, exist_ok=True)
        self.db_path = os.path.join(self.simulation_folder, "evolution.db")
        init_database(self.db_path)

        # Sauvegarde de parameters.json (une seule écriture)
        chemin_json = os.path.join(self.simulation_folder, "parameters.json")
        with open(chemin_json, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4)
        print(f"✅ Paramètres sauvegardés dans : {chemin_json}")

        # --- INITIALISATION PYGAME ---
        self.headless = headless
        self.max_steps = max_steps
        # Panneau latéral « cerveau » : ~1/4 de la largeur du monde
        self.panel_width = int(self.width // 4)
        pygame.init()
        self.screen = pygame.display.set_mode((int(self.width + self.panel_width), int(self.height + DASHBOARD_SIZE)))
        self.overlay = pygame.Surface((int(self.width), int(self.height + DASHBOARD_SIZE)), pygame.SRCALPHA)
        pygame.display.set_caption("Simulation SBN Evolution")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.font_title = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 12)
        self.font_tiny = pygame.font.SysFont("Arial", 10)
        # Agent dont le cerveau est affiché dans le panneau latéral (None = panneau fermé)
        self.tracked_agent = None

        # --- POPULATION INITIALE ---
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(
                i,
                random.randint(0, self.width),
                random.randint(0, self.height),
                self.base_energy,
                self.rotate_deg,
                self.width,
                self.height,
                self.cost_rotate,
                self.cost_move,
                self.cost_eat,
                self.cost_neuron,
                self.cost_metabolism
            )
            self.agents.append(agent)

            # Sauvegarde le réseau initial (snapshot)
            save_network_state(
                db_path=self.db_path,
                agent_id=agent.id,
                parent_agent_id=None,
                sb_network=agent.sbn,
                iteration=0,
                is_snapshot=True,
                message="Agent initial"
            )

        # --- NOURRITURE INITIALE ---
        self.foods = []
        for _ in range(self.num_food):
            x, y = random.randint(0, self.width), random.randint(0, self.height)
            # La nourriture initiale est répartie sur la grille (centre + voisins) et active immédiatement
            for (dx, dy), energie_part in self.diffusion_grid.compute_deposits(x, y, 500):
                self.foods.append(Food(dx, dy, energy=energie_part, active=True))

        # --- ÉTAT DE LA SIMULATION ---
        self.running = True
        self.show_graphics = not headless  # Variable permettant d'afficher ou non la simulation (pour aller plus vite si nécessaire)
        self.total_steps = 0  # Variable stockant le nombre de pas de la simulation
        self.new_id = self.num_agents  # Nouveau id à incrémenter à partir du nombre d'agents initiaux (pour la descendance)
        self.is_paused = False  # Variable permettant de mettre en pause la simulation
        self.temps_simule_ms = 0
        self.vision_cone = True

        # Grille spatiale de recherche de voisins
        self.spatial_grid = SpatialGrid(self.width, self.height, self.cell_size)

        # On pré calcule le carré de la distance pour éviter les racines carrées (LENT).
        self.dist_manger_sq = self.distance_manger * self.distance_manger
        self.distance_vision_sq = self.distance_vision ** 2

        # Listes pour stocker l'historique
        self.stats_steps = []
        self.stats_pop = []
        self.stats_size = []
        self.stats_energy = []
        self.stats_node_activated = []
        self.stats_global_energy = []

        self.digestion_calendar = {}  # Calendrier des déchets à activer (fil d'attente)

    # ------------------------------------------------------------------
    # Événements clavier / souris
    # ------------------------------------------------------------------
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.headless:
                    # En headless il n'y a ni fenêtre ni utilisateur : le pilote SDL
                    # "dummy" poste parfois un événement QUIT spontanément, qui ne doit
                    # pas interrompre la simulation (le seul arrêt légitime est MAX_ITERATIONS).
                    continue
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Espace pour faire Pause/play
                    self.is_paused = not self.is_paused
                    if not self.show_graphics:
                        show_graphics_off(self.screen, self.font, self.width, self.height, self.is_paused, DASHBOARD_SIZE)
                elif event.key == pygame.K_g:  # Touche G pour activer/désactiver le rendu
                    self.show_graphics = not self.show_graphics
                    if not self.show_graphics:
                        show_graphics_off(self.screen, self.font, self.width, self.height, self.is_paused, DASHBOARD_SIZE)
                elif event.key == pygame.K_v:  # Touche v pour activer/désactiver le rendu graphique des cones de vision
                    self.vision_cone = not self.vision_cone
                elif event.key == pygame.K_s:
                    show_simulation_summary(self.stats_steps, self.stats_pop, self.stats_size, self.stats_energy,
                                            self.stats_node_activated, self.stats_global_energy, None)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 1 correspond au clic gauche
                    mx, my = pygame.mouse.get_pos()
                    # Clic dans le panneau latéral : ignoré (la souris ne gère que la zone de simulation)
                    if mx >= self.width:
                        continue
                    self.is_paused = True
                    # On ramène le clic de la souris dans le référentiel de la simulation
                    my = my - DASHBOARD_SIZE

                    rayon_sq = self.taille_agent * self.taille_agent
                    # On cherche quel agent a été cliqué
                    clicked = None
                    for agent in self.agents:
                        if not agent.alive:
                            continue
                        dx = mx - agent.x
                        dy = my - agent.y

                        # Test de collision Cercle / Point sans racine carrée
                        if (dx * dx + dy * dy) <= rayon_sq:
                            clicked = agent
                            # On a trouvé l'agent cliqué, inutile de tester les autres
                            break
                    # Clic sur un agent : son cerveau s'affiche dans le panneau latéral ;
                    # clic sur le vide : le panneau se ferme.
                    self.tracked_agent = clicked
                    if clicked is not None:
                        self.tracking_id = clicked.id

    # ------------------------------------------------------------------
    # Logique : un pas de simulation
    # ------------------------------------------------------------------
    def step(self):
        self.total_steps += 1
        new_enfants = []

        if self.total_steps % 50000 == 0:
            save_nodes_influence(self.agents, self.total_steps, self.simulation_folder)

            # Snapshots périodiques pour l'évolution des réseaux
            for agent in self.agents:
                save_network_state(
                    db_path=self.db_path,
                    agent_id=agent.id,
                    parent_agent_id=agent.parent.id if agent.parent else None,
                    sb_network=agent.sbn,
                    iteration=self.total_steps,
                    is_snapshot=True,
                    message="Snapshot périodique"
                )

        # Activations des déchets
        if self.total_steps in self.digestion_calendar:
            for waste_to_activate in self.digestion_calendar[self.total_steps]:
                waste_to_activate.active = True
            # On vide la mémoire du calendrier pour ce tour
            del self.digestion_calendar[self.total_steps]

        # Mélanger pour l'équité
        random.shuffle(self.agents)

        eat_intentions = {}  # Dictionnaire stockant l'intention de manger de chaque agent

        self.spatial_grid.rebuild(self.agents, self.foods)

        for agent in self.agents:
            if not agent.alive:
                continue

            # On récupère la liste des entités proches de nous
            neighbors = self.spatial_grid.neighbors(agent)
            # On enregistre dans agent.proie_potentielle si une cible est à sa portée
            agent.sense(neighbors, self.distance_vision_sq, self.dist_manger_sq, self.vision_angle)

            # Décision manger
            action_eat = agent.update(agent.vision_input, self.proba_evolution, self.valeur_max_poids)

            # Manger
            # On utilise la proie qu'on a trouvée dans la boucle de vision
            if action_eat and (agent.proie_potentielle is not None):
                victim = agent.proie_potentielle
                # Vérification de sécurité
                if victim.alive:
                    if isinstance(victim, Agent):
                        eat_intentions[agent] = agent.proie_potentielle
                    elif isinstance(victim, Food):
                        agent.eat(victim)
                        # La nourriture mangée est balayée en fin de pas (mark-and-sweep)

            # Digestion
            if agent.stomach > self.digestion_min:
                waste = agent.digestion()

                target_step = self.total_steps + self.digestion_interval
                if target_step not in self.digestion_calendar:
                    self.digestion_calendar[target_step] = []

                # Les déchets sont répartis sur la grille (centre + voisins), inactifs jusqu'à activation
                for (dx, dy), energie_part in self.diffusion_grid.compute_deposits(agent.x, agent.y, waste):
                    new_food = Food(dx, dy, energy=energie_part, active=False)
                    self.foods.append(new_food)
                    self.digestion_calendar[target_step].append(new_food)

            if agent.energy >= self.division_energy:
                enfant = agent.division(self.new_id, self.vision_angle, self.proba_deletion, self.proba_insertion,
                                        self.valeur_max_poids, self.total_steps)
                self.new_id += 1
                new_enfants.append(enfant)  # l'enfant est ajouté à la fin du pas (il n'agit pas immédiatement)

                # ✅ Sauvegarder UNIQUEMENT l'enfant après mutation
                save_network_state(
                    db_path=self.db_path,
                    agent_id=enfant.id,
                    parent_agent_id=agent.id,
                    sb_network=enfant.sbn,
                    iteration=self.total_steps,
                    is_snapshot=False,
                    message="Mutation à la naissance"
                )

        # Vérification de conflit lors de l'action "manger"
        for agent, victim in eat_intentions.items():
            # On vérifie que l'agent et la victime sont bien toujours vivant
            if not agent.alive or not victim.alive:
                continue

            # CAS 1 : Duel
            if eat_intentions.get(victim) == agent:
                if agent.energy > victim.energy:
                    agent.eat(victim)
                elif agent.energy == victim.energy:
                    if random.random() > 0.5:
                        agent.eat(victim)
                    else:
                        victim.eat(agent)
                else:
                    victim.eat(agent)
            # CAS 2 : Normal
            else:
                agent.eat(victim)

        # Nettoyage rapide
        if self.active_corpse:
            for a in self.agents:
                # On cible ceux qui viennent de mourir (famine ou vieillesse)
                # et qui n'ont pas été "vidés" par un prédateur.
                if not a.alive and a.stomach + a.energy != 0:
                    # On dépose le contenu de leur estomac au sol, réparti sur la grille
                    agent_rest = a.stomach + a.energy
                    # On vide l'estomac pour éviter les doublons si le code repasse dessus
                    a.stomach = 0

                    target_step = self.total_steps + self.digestion_interval
                    if target_step not in self.digestion_calendar:
                        self.digestion_calendar[target_step] = []

                    for (dx, dy), energie_part in self.diffusion_grid.compute_deposits(a.x, a.y, agent_rest):
                        new_food = Food(dx, dy, energy=energie_part, active=False)
                        self.foods.append(new_food)
                        # On l'ajoute au calendrier
                        self.digestion_calendar[target_step].append(new_food)

        # Ensuite, on fait le nettoyage habituel (mark-and-sweep : un seul parcours
        # par liste au lieu d'un list.remove() O(n) à chaque repas)
        self.agents = [a for a in self.agents if a.alive]
        self.foods = [f for f in self.foods if f.alive]
        self.agents.extend(new_enfants)

    # ------------------------------------------------------------------
    # Rendu
    # ------------------------------------------------------------------
    def render(self):
        self.screen.fill((0, 0, 0))
        # On le vide à chaque frame
        self.overlay.fill((0, 0, 0, 0))

        # Affichage des agents
        for agent in self.agents:
            is_tracked = (agent.id == self.tracking_id)
            if self.distance_vision > 20:  # On applique l'opacité sur le cone de vision pour plus de visibilité (si le cone est assez grand sinon on le met en blanc, car il ne sera pas visible).
                agent.draw(self.screen, self.overlay, self.taille_agent, self.distance_vision, self.vision_angle,
                           self.base_energy, DASHBOARD_SIZE,
                           vision_cone=self.vision_cone,
                           tracking=is_tracked)  # On applique l'opacité sur le cone de vision pour plus de visibilité
            else:
                agent.draw(self.screen, self.screen, self.taille_agent, self.distance_vision, self.vision_angle,
                           self.base_energy, DASHBOARD_SIZE,
                           vision_cone=self.vision_cone, tracking=is_tracked)

        # NOTE : la nourriture inactive (déchets/cadavres en attente d'activation) est
        # directement visible via Food.draw() (cercle marron), la grille de diffusion
        # ne sert plus qu'au calcul de répartition, plus à l'affichage.
        for food in self.foods:
            food.draw(self.screen, DASHBOARD_SIZE)  # Affiche chaque particule de nourriture (active ou inactive)
        # On fusionne le calque avec le screen
        self.screen.blit(self.overlay, (0, 0))

        # On calcule le nombre de neurones en moyenne
        pop = len(self.agents)
        if pop > 0:
            mean_nodes = sum(a.sbn.num_nodes for a in self.agents) / pop
        else:
            mean_nodes = 0.0
        # On affiche le dashboard
        draw_dashboard(self.screen, self.clock, pop, self.total_steps, mean_nodes, PARAMS, self.width,
                       DASHBOARD_SIZE, self.font, self.temps_simule_ms)

        if self.is_paused:
            graphics_pause(self.screen, self.font, DASHBOARD_SIZE)

        # Panneau latéral : cerveau de l'agent suivi (intégré à la fenêtre, pas de fenêtre séparée)
        self.draw_sbn_panel()

        # On rafraîchit l'écran une fois après la boucle des agents
        pygame.display.flip()
        self.clock.tick(self.fps)

    # ------------------------------------------------------------------
    # Rendu : panneau latéral « cerveau » de l'agent suivi
    # ------------------------------------------------------------------
    def draw_sbn_panel(self):
        """Dessine le SBN de l'agent suivi dans le panneau latéral (~1/4 de la
        largeur de la fenêtre). Remplace l'ancienne fenêtre matplotlib séparée
        (bloquante, impossible à fermer). Clic sur un agent = suivi ;
        clic sur le vide = panneau fermé."""
        agent = self.tracked_agent
        if agent is None:
            return

        panel_x = self.width
        panel_y = DASHBOARD_SIZE
        panel_w = self.panel_width
        panel_h = self.height

        # Fond et bordure gauche
        pygame.draw.rect(self.screen, (10, 14, 22), (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.line(self.screen, (90, 110, 140), (panel_x, panel_y), (panel_x, panel_y + panel_h), 2)

        # En-tête + informations de l'agent
        self.screen.blit(self.font_title.render(f"Cerveau agent {agent.id}", True, (235, 240, 250)),
                         (panel_x + 10, panel_y + 8))
        alive = agent.alive
        info = [
            ("État: " + ("vivant" if alive else "MORT (réseau gelé)"),
             (120, 220, 140) if alive else (240, 120, 120)),
            (f"Noeuds: {agent.sbn.num_nodes}   Âge: {agent.step} pas", (165, 180, 205)),
            (f"Énergie: {agent.energy:.0f}  Estomac: {agent.stomach:.0f}", (165, 180, 205)),
        ]
        if agent.parent is not None:
            info.append((f"Parent: {agent.parent.id}", (165, 180, 205)))
        y = panel_y + 30
        for text, color in info:
            self.screen.blit(self.font_small.render(text, True, color), (panel_x + 10, y))
            y += 15
        y += 8

        # Zone du graphe
        graph_h = panel_h - (y - panel_y) - 30
        cx = panel_x + panel_w / 2
        cy = y + graph_h / 2
        radius = max(28.0, min(panel_w, graph_h) / 2 - 26)

        sbn = agent.sbn
        n = sbn.num_nodes
        if n == 0:
            self.screen.blit(self.font_small.render("Aucun nœud", True, (150, 160, 180)), (panel_x + 10, y + 20))
            return

        # Positions des nœuds : rangement circulaire (ordre du réseau, étiquette = true id)
        node_radius = max(8, min(14, int(110 / n)))
        pos = []
        for i in range(n):
            ang = -math.pi / 2 + 2 * math.pi * i / n
            pos.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))

        # Étiquettes (mêmes noms que l'ancienne fenêtre matplotlib)
        NAMES = {0: "E", 1: "M", 2: "F", 3: "R", 4: "R1", 5: "R2", 6: "R3", 7: "R4"}
        labels = [NAMES.get(sbn.true_ids[i], f"N{sbn.true_ids[i]}") for i in range(n)]
        is_organ = [sbn.true_ids[i] < 4 for i in range(n)]

        # 1) Arêtes (sous les nœuds) : vert = poids > 0, rouge = poids < 0, épaisseur ∝ |poids|
        for i in range(n):
            for j in range(n):
                w = int(sbn.weights[i, j])
                if w == 0:
                    continue
                color = (80, 200, 90) if w > 0 else (225, 75, 75)
                e_width = min(1 + abs(w), 4)
                if i == j:
                    # Boucle : petit cercle sous le nœud (au-dessus c'est l'étiquette)
                    nx_, ny_ = pos[i]
                    pygame.draw.circle(self.screen, color, (int(nx_), int(ny_ + node_radius + 10)), 5, e_width)
                else:
                    x1, y1 = pos[i]
                    x2, y2 = pos[j]
                    dx, dy = x2 - x1, y2 - y1
                    dist = math.hypot(dx, dy)
                    if dist < node_radius * 2 + 4:
                        continue  # nœuds trop proches pour dessiner l'arête
                    ux, uy = dx / dist, dy / dist
                    sx = x1 + ux * (node_radius + 2)
                    sy = y1 + uy * (node_radius + 2)
                    ex = x2 - ux * (node_radius + 7)
                    ey = y2 - uy * (node_radius + 7)
                    pygame.draw.line(self.screen, color, (sx, sy), (ex, ey), e_width)
                    # Flèche à l'arrivée
                    ang = math.atan2(ey - sy, ex - sx)
                    fl = 7
                    pygame.draw.polygon(self.screen, color, [
                        (ex, ey),
                        (ex - fl * math.cos(ang - 0.45), ey - fl * math.sin(ang - 0.45)),
                        (ex - fl * math.cos(ang + 0.45), ey - fl * math.sin(ang + 0.45)),
                    ])
                    # Poids en milieu d'arête
                    label = self.font_tiny.render(str(w), True, (215, 220, 235))
                    self.screen.blit(label, (int((sx + ex) / 2 - label.get_width() / 2 + 4),
                                             int((sy + ey) / 2 - label.get_height() / 2 - 5)))

        # 2) Nœuds : organe = bleu, interne = gris ; rempli clair = état actif (1)
        for i in range(n):
            x, y2 = pos[i]
            active = int(sbn.states[i]) == 1
            if is_organ[i]:
                base = (150, 200, 255) if active else (65, 105, 175)
            else:
                base = (250, 205, 70) if active else (95, 100, 118)
            pygame.draw.circle(self.screen, base, (int(x), int(y2)), node_radius)
            if active:
                pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y2)), node_radius, 1)
            # Étiquette du nœud, au-dessus
            label = self.font_tiny.render(labels[i], True,
                                          (235, 240, 250) if is_organ[i] else (185, 195, 215))
            self.screen.blit(label, (int(x - label.get_width() / 2),
                                     int(y2 - node_radius - label.get_height() - 3)))

        # Légende
        legend = "clair = actif ; vert = + ; rouge = -"
        s = self.font_tiny.render(legend, True, (140, 150, 170))
        self.screen.blit(s, (int(cx - s.get_width() / 2), panel_y + panel_h - 16))

    # ------------------------------------------------------------------
    # Statistiques
    # ------------------------------------------------------------------
    def collect_stats(self):
        if self.total_steps % 100 == 0 and len(self.agents) > 0:
            self.stats_steps.append(self.total_steps)
            self.stats_pop.append(len(self.agents))
            age_max = max(a.step for a in self.agents)
            energy_agent = sum(a.energy + a.stomach for a in self.agents)
            energy_sol = sum(f.energy for f in self.foods)
            energy_total = energy_agent + energy_sol

            # Calcul des moyennes
            moyenne_noeuds = sum(a.sbn.num_nodes for a in self.agents) / len(self.agents)
            moyenne_energie = sum(a.energy for a in self.agents) / len(self.agents)
            moyenne_noeuds_actif = sum(float(np.sum(a.sbn.states)) for a in self.agents) / len(self.agents)
            self.stats_size.append(moyenne_noeuds)
            self.stats_energy.append(moyenne_energie)
            self.stats_node_activated.append(moyenne_noeuds_actif)
            self.stats_global_energy.append(energy_total)

            if not self.show_graphics:
                secondes_ecoulees = self.temps_simule_ms // 1000
                fps_actuel = self.clock.get_fps()
                print(
                    f"[{secondes_ecoulees}s] Step: {self.total_steps} | Pop: {len(self.agents)} | Énergie Moy: {moyenne_energie:.0f} | Noeuds Moy: {moyenne_noeuds:.1f} | Âge Max: {age_max}")

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------
    def run(self):
        while self.running:
            # 1. Paramétrage des touches de la simulation
            self.handle_events()

            # 2. LOGIQUE
            if not self.is_paused:
                for _ in range(self.n_iter):
                    t_start = self.clock.get_time()
                    self.step()
                    self.collect_stats()
                    # Temps réellement simulé : delta du pas (et non le temps cumulé depuis le start)
                    self.temps_simule_ms += self.clock.get_time() - t_start

                    if self.max_steps and self.total_steps >= self.max_steps:
                        self.running = False

            # 3. AFFICHAGE (RENDU)
            if self.show_graphics and not self.headless:
                self.render()

        self.close()

    def close(self):
        """Sauvegarde des statistiques, fermeture de la base et de pygame."""
        # Historique des statistiques (exploitable pour l'analyse)
        stats_path = os.path.join(self.simulation_folder, "stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_steps": self.total_steps,
                "final_pop": len(self.agents),
                "final_foods": len(self.foods),
                "stats_steps": self.stats_steps,
                "stats_pop": self.stats_pop,
                "stats_size": self.stats_size,
                "stats_energy": self.stats_energy,
                "stats_node_activated": self.stats_node_activated,
                "stats_global_energy": self.stats_global_energy
            }, f)
        print(f"✅ Statistiques sauvegardées dans : {stats_path}")

        close_all_connections()
        pygame.quit()

        summary_save_folder = os.path.join(self.simulation_folder, "simulation_summary.png")
        show_simulation_summary(self.stats_steps, self.stats_pop, self.stats_size, self.stats_energy,
                                self.stats_node_activated, self.stats_global_energy, summary_save_folder)


def _coerce_param(value):
    """Convertit la valeur d'un --param CLE=VALEUR en int, float, bool ou str."""
    v = value.strip()
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_args():
    parser = argparse.ArgumentParser(description="Simulation SBN — écosystème évolutionnaire")
    parser.add_argument("--headless", action="store_true",
                        help="Pas de fenêtre (SDL en mode dummy), exécution en tâche de fond")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Arrêt automatique après N pas (surcharge PARAMS['MAX_ITERATIONS'] ; non passé = valeur de PARAMS)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Surcharge la graine aléatoire PARAMS['SEED']")
    parser.add_argument("--param", action="append", default=[], metavar="CLE=VALEUR",
                        help="Surcharge un paramètre PARAMS (répétable, ex. --param NUM_AGENTS=300)")
    parser.add_argument("--outdir", default=None,
                        help="Nom du sous-dossier dans simulations/ (défaut : horodatage)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        PARAMS["SEED"] = args.seed
    for spec in args.param:
        if "=" not in spec:
            sys.exit(f"Format --param invalide : {spec!r} (attendu CLE=VALEUR)")
        key, _, value = spec.partition("=")
        key = key.strip()
        if key not in PARAMS:
            sys.exit(f"Paramètre inconnu : {key!r} (clés disponibles : {', '.join(sorted(PARAMS))})")
        PARAMS[key] = _coerce_param(value)
    # Temps maximal de simulation : --max-steps (CLI) prioritaire sur PARAMS["MAX_ITERATIONS"]
    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        max_steps = int(PARAMS.get("MAX_ITERATIONS", 0))
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    simulation = Simulation(PARAMS, headless=args.headless, max_steps=max_steps, outdir=args.outdir)
    simulation.run()


if __name__ == "__main__":
    main()
