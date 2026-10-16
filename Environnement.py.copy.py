import pygame
import random
import math
import json
import os
import numpy as np
from Agent import Agent
from GraphVisualisation import *
from Interface import *
from SpatialGrid import *
from Food import Food
from ExportData import *
import sqlite3
from evolution_tracker import save_network_state
from DiffusionGrid import DiffusionGrid
from datetime import datetime
from simulation_setup import setup_simulation

# =============================================
# INITIALISATION DE LA BASE DE DONNÉES (DOIT ÊTRE EN PREMIER)
# =============================================
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# SIMULATION_DIR = f"simulations/sim_{timestamp}"
# os.makedirs(SIMULATION_DIR, exist_ok=True)
# DB_PATH = os.path.join(SIMULATION_DIR, "evolution.db")

# Utilise TEST_NAME + timestamp pour un nom unique (v2)
# =============================================
# PARAMÈTRES DE LA SIMULATION (DOIT ÊTRE AVANT TOUT LE RESTE)
# =============================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SIMULATION_FOLDER = os.path.join("simulations", timestamp)
os.makedirs(SIMULATION_FOLDER, exist_ok=True)
DB_PATH = os.path.join(SIMULATION_FOLDER, "evolution.db")

# 🔹 CRÉE LES TABLES SI ELLES N'EXISTENT PAS
conn = sqlite3.connect(DB_PATH)
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
conn.commit()
conn.close()

# =============================================
# PARAMÈTRES DE LA SIMULATION
# =============================================
PARAMS = {
    "TEST_NAME": "No_weights_cap_higher_cost_neuron",  # (Garde-le si besoin, mais il ne sert plus pour le nom du dossier)
    "SEED": 42,
    # ... (le reste de tes paramètres)
}

# Sauvegarde de parameters.json dans SIMULATION_FOLDER
chemin_json = os.path.join(SIMULATION_FOLDER, "parameters.json")
with open(chemin_json, "w", encoding="utf-8") as f:
    json.dump(PARAMS, f, indent=4)
print(f"✅ Paramètres sauvegardés dans : {chemin_json}")

# Paramètre de la simulation
PARAMS = {
    "TEST_NAME": "No_weights_cap_higher_cost_neuron",
    "SEED": 42,
    "NUM_AGENTS": 150,
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

# On récupère les paramètres du dictionnaire s'ils existent sinon on met les valeurs par défauts
# --- PARAMÈTRES GÉNÉRAUX ---
TEST_NAME = str(PARAMS.get("TEST_NAME", "Extinction_Initiale"))  # Nom de la session ou de l'expérience en cours
SEED = PARAMS.get("SEED", 42)  # Graine de génération aléatoire (assure que la simulation est reproductible)

# --- MOTEUR ET AFFICHAGE ---
WIDTH = int(PARAMS.get("WIDTH", 1280))  # Largeur de la fenêtre de simulation en pixels
HEIGHT = int(PARAMS.get("HEIGHT", 720))  # Hauteur de la fenêtre de simulation en pixels
N_ITER = int(PARAMS.get("N_ITER",
                        1))  # Nombre de mises à jour logiques par image (accélère le temps simulé sans affecter l'affichage)
FPS = int(PARAMS.get("FPS", 60))  # Limite d'images par seconde pour le rendu visuel
DISPLAY_FPS = PARAMS.get("DISPLAY_FPS", 0)  # Active (1) ou désactive (0) l'affichage du compteur visuel des FPS

# --- PHYSIQUE ET CAPACITÉS DES AGENTS ---
TAILLE_AGENT = int(PARAMS.get("TAILLE_AGENT", 5))  # Rayon d'affichage du cercle représentant l'agent
ROTATE_DEG = PARAMS.get("ROTATE_DEG",
                        10)  # Vitesse de rotation : nombre de degrés modifiés quand l'agent décide de tourner
VISION_ANGLE = PARAMS.get("VISION_ANGLE", 45)  # Ouverture du cône de vision de l'agent (en degrés)
DISTANCE_MANGER = int(
    PARAMS.get("DISTANCE_MANGER", TAILLE_AGENT * 2))  # Portée maximale à laquelle l'agent peut attaquer/manger
DISTANCE_VISION = int(
    PARAMS.get("DISTANCE_VISION", DISTANCE_MANGER))  # Portée maximale à laquelle l'agent détecte les autres entités

# --- DYNAMIQUE DE POPULATION ET ÉNERGIE ---
NUM_AGENTS = int(PARAMS.get("NUM_AGENTS", 1000))  # Taille de la population de départ générée aléatoirement
BASE_ENERGY = PARAMS.get("BASE_ENERGY", 100)  # Capital énergétique de départ pour les premiers agents
DIVISION_ENERGY = PARAMS.get("DIVISION_ENERGY",
                             600)  # Palier d'énergie requis pour déclencher la reproduction par division
NUM_FOOD = int(PARAMS.get("NUM_FOOD", 100))
COST_MOVE = PARAMS.get("COST_MOVE", 1)
COST_ROTATE = PARAMS.get("COST_ROTATE", 1)
COST_EAT = PARAMS.get("COST_EAT", 1)
COST_NEURON = PARAMS.get("COST_NEURON", 1)
COST_METABOLISM = PARAMS.get("COST_METABOLISM", 0.05)
DIGESTION_MIN = PARAMS.get("DIGESTION_MIN", 40)
DIGESTION_RATE = PARAMS.get("DIGESTION_RATE", 5)
DIGESTION_INTERVAL = PARAMS.get("DIGESTION_INTERVAL", 10)
ACTIVE_CORPSE = PARAMS.get("ACTIVE_CORPSE", True)
DIFFUSION_RATE = PARAMS.get("DIFFUSION_RATE", 2/3)
SEUIL_DIFFUSION = PARAMS.get("SEUIL_DIFFUSION", 300)

# --- MUTATIONS DU CERVEAU (RÉSEAU DE NEURONES) ---
PROBA_DELETION = PARAMS.get("PROBA_DELETION",
                            0.01)  # Probabilité qu'un agent perde un nœud neuronal lors d'une mutation
PROBA_INSERTION = PARAMS.get("PROBA_INSERTION", 0.02)  # Probabilité qu'un agent développe un nouveau nœud neuronal
PROBA_EVOLUTION = PARAMS.get("PROBA_EVOLUTION", 0.02)  # Probabilité qu'un agent développe un nouveau nœud neuronal
VALEUR_MAX_POIDS = PARAMS.get("VALEUR_MAX_POIDS", 3)  # Valeur absolue maximale des connexions créées entre les neurones

# --- STATISTIQUE ET VISUALISATION ---
TRACKING_ID = PARAMS.get("TRACKING_ID", None)

# Paramètre de la grille spatiale (recherche de voisins, indépendante de la diffusion de nourriture)
CELL_SIZE = DISTANCE_VISION * 1.2  # Doit être >= DISTANCE_VISION (on met distance_vision + 20% pour avoir de la marge d'erreur)

# Grille de diffusion de nourriture : résolution fine, basée sur la taille d'un agent, indépendante de CELL_SIZE
FOOD_CELL_SIZE = TAILLE_AGENT
diffusion_grid = DiffusionGrid(WIDTH, HEIGHT, FOOD_CELL_SIZE, DIFFUSION_RATE, SEUIL_DIFFUSION)  # diffusion nourriture


random.seed(
    SEED)  # On applique la seed à random (cela marchera peu importe ou random est appelé (n'importe quel classe))
np.random.seed(SEED)  # Pareil pour le random de numpy

# Dashboard d'information
DASHBOARD_SIZE = 50

# # --- SIMULATION SAVE ---
# # Création du dossier de sauvegarde et sauvegarde des paramètres
# # --- INITIALISATION DES CHEMINS ---
# BASE_TEST_NAME = TEST_NAME
# SIMULATION_FOLDER = os.path.join("simulations", BASE_TEST_NAME)
#
# # Gestion des noms uniques (si le dossier existe déjà)
# if os.path.exists(SIMULATION_FOLDER):
#     count = 1
#     while os.path.exists(os.path.join("simulations", f"{BASE_TEST_NAME}_{count}")):
#         count += 1
#     TEST_NAME = f"{BASE_TEST_NAME}_{count}"
#     SIMULATION_FOLDER = os.path.join("simulations", TEST_NAME)
#
# PARAMS["TEST_NAME"] = TEST_NAME
#
# # Création du dossier de simulation
# os.makedirs(SIMULATION_FOLDER, exist_ok=True)
#
# # Sauvegarde de parameters.json dans le dossier de simulation
# chemin_json = os.path.join(SIMULATION_FOLDER, "parameters.json")
# with open(chemin_json, "w", encoding="utf-8") as f:
#     json.dump(PARAMS, f, indent=4)
# print(f"✅ Paramètres sauvegardés dans : {chemin_json}")
#
# # Chemin pour evolution.db
# db_path = os.path.join(SIMULATION_FOLDER, "evolution.db")
#
# # On sauvegarde le dictionnaire dans un fichier parameters.json
# chemin_json = os.path.join(SIMULATION_FOLDER, "parameters.json")
# with open(chemin_json, "w", encoding="utf-8") as f:
#     json.dump(PARAMS, f, indent=4)
#
# print(f"✅ Paramètres sauvegardés dans : {chemin_json}")  # Debug
# with open(chemin_json, "w", encoding="utf-8") as f:
#     json.dump(PARAMS, f, indent=4)


# Sauvegarde de parameters.json dans SIMULATION_FOLDER
chemin_json = os.path.join(SIMULATION_FOLDER, "parameters.json")
with open(chemin_json, "w", encoding="utf-8") as f:
    json.dump(PARAMS, f, indent=4)
print(f"✅ Paramètres sauvegardés dans : {chemin_json}")


# --- INITIALISATION PYGAME ---
pygame.init()
screen = pygame.display.set_mode((int(WIDTH), int(HEIGHT + DASHBOARD_SIZE)))
overlay = pygame.Surface((int(WIDTH), int(HEIGHT + DASHBOARD_SIZE)), pygame.SRCALPHA)
pygame.display.set_caption("Simulation SBN Evolution")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 18)

# Liste stockant les agents
agents = []
for i in range(NUM_AGENTS):
    agent = Agent(
        i,
        random.randint(0, WIDTH),
        random.randint(0, HEIGHT),
        BASE_ENERGY,
        ROTATE_DEG,
        WIDTH,
        HEIGHT,
        COST_ROTATE,
        COST_MOVE,
        COST_EAT,
        COST_NEURON,
        COST_METABOLISM
    )
    agents.append(agent)

    # Sauvegarde le réseau initial
    save_network_state(
        db_path=DB_PATH,
        agent_id=agent.id,
        parent_agent_id=None,
        sb_network=agent.sbn,
        iteration=0,
        is_snapshot=True,
        message="Agent initial"
    )
# Ajoute ceci pour sauvegarder les agents initiaux
for agent in agents:
    save_network_state(
        db_path=DB_PATH,
        agent_id=agent.id,          # ID unique de l'agent
        parent_agent_id=None,      # Pas de parent pour les agents initiaux
        sb_network=agent.sbn,      # Son réseau neuronal (SbNetwork)
        iteration=0,               # Itération 0 (début de la simulation)
        is_snapshot=True,          # C'est un snapshot (état complet)
        message="Agent initial"     # Message descriptif
    )
foods = []
for _ in range(NUM_FOOD):
    x, y = random.randint(0, WIDTH), random.randint(0, HEIGHT)
    # La nourriture initiale est répartie sur la grille (centre + voisins) et active immédiatement
    for (dx, dy), energie_part in diffusion_grid.compute_deposits(x, y, 500):
        foods.append(Food(dx, dy, energy=energie_part, active=True))
running = True
show_graphics = True  # Variable permettant d'afficher ou non la simulation (pour aller plus vite si nécessaire)
total_steps = 0  # Variable stockant le nombre de pas de la simulation

new_id = NUM_AGENTS  # Nouveau id à incrémenter à partir du nombre d'agents initiaux (pour la descendance)

is_paused = False  # Variable permettant de mettre en pause la simulation
temps_simule_ms = 0

vision_cone = True

# On prés calcule le carré de la distance pour éviter les racines carrées (LENT).
DIST_MANGER_SQ = DISTANCE_MANGER * DISTANCE_MANGER
DISTANCE_VISION_SQ = DISTANCE_VISION ** 2

# Listes pour stocker l'historique
stats_steps = []
stats_pop = []
stats_size = []
stats_energy = []
stats_node_activated = []
stats_global_energy = []

digestion_calendar = {}  # Calendrier des déchets à activer (fil d'attente)

while running:
    # 1. Paramétrage des touches de la simulation
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # Espace pour faire Pause/play
                is_paused = not is_paused
                if not show_graphics: show_graphics_off(screen, font, WIDTH, HEIGHT, is_paused, DASHBOARD_SIZE)
            elif event.key == pygame.K_g:  # Touche G pour activer/désactiver le rendu
                show_graphics = not show_graphics
                if not show_graphics: show_graphics_off(screen, font, WIDTH, HEIGHT, is_paused, DASHBOARD_SIZE)
            elif event.key == pygame.K_v:  # Touche v pour activer/désactiver le rendu graphique des cones de vision
                vision_cone = not vision_cone
            elif event.key == pygame.K_s:
                show_simulation_summary(stats_steps, stats_pop, stats_size, stats_energy, stats_node_activated,
                                        stats_global_energy, None)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 1 correspond au clic gauche
                is_paused = True
                mx, my = pygame.mouse.get_pos()
                # On ramène le clic de la souris dans le référentiel de la simulation
                my = my - DASHBOARD_SIZE

                rayon_sq = TAILLE_AGENT * TAILLE_AGENT
                # On cherche quel agent a été cliqué
                for agent in agents:
                    if not agent.alive: continue
                    dx = mx - agent.x
                    dy = my - agent.y

                    # Test de collision Cercle / Point sans racine carrée
                    if (dx * dx + dy * dy) <= rayon_sq:
                        # Appel de la fonction de visualisation du réseau sbn
                        show_sbn_graph(agent.id, agent.sbn)
                        # On a trouvé l'agent cliqué, inutile de tester les autres
                        break

    # 2. LOGIQUE
    if not is_paused:
        for _ in range(N_ITER):
            total_steps += 1
            temps_simule_ms += clock.get_time()
            new_enfants = []

            if total_steps % 50000 == 0:
                save_nodes_influence(agents, total_steps, SIMULATION_FOLDER)

                # NOUVEAU : Snapshots périodiques pour l'évolution des réseaux
                for agent in agents:
                    save_network_state(
                        db_path=DB_PATH,
                        agent_id=agent.id,
                        parent_agent_id=agent.parent.id if agent.parent else None,
                        sb_network=agent.sbn,
                        iteration=total_steps,
                        is_snapshot=True,
                        message="Snapshot périodique"
                    )
            # Activations des déchets
            if total_steps in digestion_calendar:
                for waste_to_activate in digestion_calendar[total_steps]:
                    waste_to_activate.active = True
                # On vide la mémoire du calendrier pour ce tour
                del digestion_calendar[total_steps]

            # Mélanger pour l'équité
            random.shuffle(agents)

            eat_intentions = {}  # Dictionnaire stockant l'intention de manger de chaque agent

            spatial_grid = update_grid(agents, foods, CELL_SIZE)

            for agent in agents:
                if not agent.alive:
                    continue

                # On récupère la liste des cases proches de nous
                neighbors = get_neighbors(agent, spatial_grid, CELL_SIZE)
                # On enregistre dans agent.proie_potentielle si une cible est à sa portée
                agent.sense(neighbors, DISTANCE_VISION_SQ, DIST_MANGER_SQ, VISION_ANGLE)

                # Décision manger
                action_eat = agent.update(agent.vision_input, PROBA_EVOLUTION, VALEUR_MAX_POIDS)

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
                            foods.remove(victim)  # On enlève la nourriture du sol
                            # foods.append(Food(random.randint(0, WIDTH), random.randint(0, HEIGHT))) # On la fais réapparaître ailleurs

                # Digestion
                if agent.stomach > DIGESTION_MIN:
                    waste = agent.digestion()

                    target_step = total_steps + DIGESTION_INTERVAL
                    if target_step not in digestion_calendar:
                        digestion_calendar[target_step] = []

                    # Les déchets sont répartis sur la grille (centre + voisins), inactifs jusqu'à activation
                    for (dx, dy), energie_part in diffusion_grid.compute_deposits(agent.x, agent.y, waste):
                        new_food = Food(dx, dy, energy=energie_part, active=False)
                        foods.append(new_food)
                        digestion_calendar[target_step].append(new_food)

                if agent.energy >= DIVISION_ENERGY:
                    enfant = agent.division(new_id, VISION_ANGLE, PROBA_DELETION, PROBA_INSERTION, VALEUR_MAX_POIDS,
                                            total_steps)
                    new_id += 1
                    new_enfants.append(enfant)

                    # ✅ Sauvegarder UNIQUEMENT l'enfant après mutation
                    save_network_state(
                        db_path=DB_PATH,
                        agent_id=enfant.id,
                        parent_agent_id=agent.id,
                        sb_network=enfant.sbn,
                        iteration=total_steps,
                        is_snapshot=False,
                        message="Mutation à la naissance"
                    )

            # Vérification de conflit lors de l'action "manger"
            for agent, victim in eat_intentions.items():
                # On vérifie que l'agent et la victime sont bien toujours vivant
                if not agent.alive or not victim.alive: continue

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
            if ACTIVE_CORPSE:
                for a in agents:
                    # On cible ceux qui viennent de mourir (famine ou vieillesse)
                    # et qui n'ont pas été "vidés" par un prédateur.
                    if not a.alive and a.stomach + a.energy != 0:
                        # On dépose le contenu de leur estomac au sol, réparti sur la grille
                        agent_rest = a.stomach + a.energy
                        # On vide l'estomac pour éviter les doublons si le code repasse dessus
                        a.stomach = 0

                        target_step = total_steps + DIGESTION_INTERVAL
                        if target_step not in digestion_calendar:
                            digestion_calendar[target_step] = []

                        for (dx, dy), energie_part in diffusion_grid.compute_deposits(a.x, a.y, agent_rest):
                            new_food = Food(dx, dy, energy=energie_part, active=False)
                            foods.append(new_food)
                            # On l'ajoute au calendrier
                            digestion_calendar[target_step].append(new_food)

            # Ensuite, on fait ton nettoyage habituel
            agents = [a for a in agents if a.alive]
            agents.extend(new_enfants)

    # 3. AFFICHAGE (RENDU)
    if show_graphics:
        screen.fill((0, 0, 0))
        # On le vide à chaque frame
        overlay.fill((0, 0, 0, 0))

        # Affichage des agents
        for agent in agents:
            is_tracked = (agent.id == TRACKING_ID)
            if DISTANCE_VISION > 20:  # On applique l'opacité sur le cone de vision pour plus de visibilité (si le cone est assez grand sinon on le met en blanc, car il ne sera pas visible).
                agent.draw(screen, overlay, TAILLE_AGENT, DISTANCE_VISION, VISION_ANGLE, BASE_ENERGY, DASHBOARD_SIZE,
                           vision_cone=vision_cone,
                           tracking=is_tracked)  # On applique l'opacité sur le cone de vision pour plus de visibilité
            else:
                agent.draw(screen, screen, TAILLE_AGENT, DISTANCE_VISION, VISION_ANGLE, BASE_ENERGY, DASHBOARD_SIZE,
                           vision_cone=vision_cone, tracking=is_tracked)

        # NOTE : la nourriture inactive (déchets/cadavres en attente d'activation) est
        # directement visible via Food.draw() (cercle marron), la grille de diffusion
        # ne sert plus qu'au calcul de répartition, plus à l'affichage.
        for food in foods:
            food.draw(screen, DASHBOARD_SIZE)  # Affiche chaque particule de nourriture (active ou inactive)
        # On fusionne le calque avec le screen
        screen.blit(overlay, (0, 0))
        # On calcule le nombre de neurones en moyenne
        pop = len(agents)
        if pop > 0:
            mean_nodes = sum(a.sbn.num_nodes for a in agents) / pop
        else:
            mean_nodes = 0.0
        # On affiche le dashboard
        draw_dashboard(screen, clock, pop, total_steps, mean_nodes, PARAMS, WIDTH, DASHBOARD_SIZE, font,
                       temps_simule_ms)

        if is_paused: graphics_pause(screen, font, DASHBOARD_SIZE)

        # On rafraîchit l'écran une fois après la boucle des agents
        pygame.display.flip()
        clock.tick(FPS)

    # 4. statistiques
    if total_steps % 100 == 0 and len(agents) > 0:
        stats_steps.append(total_steps)
        stats_pop.append(len(agents))
        agent_max = max(a.step for a in agents)
        age_max = max(a.step for a in agents)
        energy_agent = sum(a.energy + a.stomach for a in agents)
        energy_sol = sum(f.energy for f in foods)
        energy_total = energy_agent + energy_sol

        # Calcul des moyennes
        moyenne_noeuds = sum(a.sbn.num_nodes for a in agents) / len(agents)
        moyenne_energie = sum(a.energy for a in agents) / len(agents)
        moyenne_noeuds_actif = sum(float(np.sum(a.sbn.states)) for a in agents) / len(agents)
        stats_size.append(moyenne_noeuds)
        stats_energy.append(moyenne_energie)
        stats_node_activated.append(moyenne_noeuds_actif)
        stats_global_energy.append(energy_total)

        if not show_graphics:
            secondes_ecoulees = temps_simule_ms // 1000
            fps_actuel = clock.get_fps()
            print(
                f"[{secondes_ecoulees}s] Step: {total_steps} | Pop: {len(agents)} | Énergie Moy: {moyenne_energie:.0f} | Noeuds Moy: {moyenne_noeuds:.1f} | Âge Max: {age_max}")

pygame.quit()

summary_save_folder = os.path.join(SIMULATION_FOLDER, "simulation_summary.png")
show_simulation_summary(stats_steps, stats_pop, stats_size, stats_energy, stats_node_activated, stats_global_energy,
                        summary_save_folder)