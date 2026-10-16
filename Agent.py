import math
import pygame
import numpy as np
import random
import copy
from SbNetwork import SbNetwork


class Agent:
    # Seuil d'énergie en dessous duquel l'agent meurt
    DEATH_ENERGY = 50
    # Distance à laquelle l'enfant est placé lors de la division (en pixels)
    SPAWN_DISTANCE = 15

    def __init__(self, id_agent, x, y, energy, rotate_deg, env_width, env_height, cost_rotate, cost_move, cost_eat,
                 cost_neuron, cost_metabolism, max_cout_A=1.0, max_cout_B=1.0):
        self.id = id_agent
        self.x = x
        self.y = y
        self.energy = energy
        self.rotate_deg = rotate_deg
        self.env_width = env_width
        self.env_height = env_height
        self.sbn = SbNetwork()
        self.speed = 2
        self.angle = random.uniform(0, 360)
        self.parent = None
        self.max_cout_A = max_cout_A  # ✅ NOUVELLE LIGNE
        self.max_cout_B = max_cout_B  # ✅ NOUVELLE LIGNE
        self.cout_base = cost_neuron  # ✅ NOUVELLE LIGNE

        # Variable stockant une information liée à la simulation
        self.alive = True
        self.proie_potentielle = None
        self.vision_input = 0
        self.step = 0

        # Cout des actions
        self.cost_rotate = cost_rotate
        self.cost_move = cost_move
        self.cost_eat = cost_eat
        self.cost_neuron = cost_neuron
        self.cost_metabolism = cost_metabolism
        # Digestion
        self.stomach = 0

    def move(self):
        # On convertit l'angle en radian pour les calculs
        rad = math.radians(self.angle)

        dx = self.speed * math.cos(rad)
        dy = self.speed * math.sin(rad)

        # On crée des nouvelles positions
        new_x = self.x + dx
        new_y = self.y + dy

        # On met à jour les nouvelles positions pour un environnement en tore
        self.x = new_x % self.env_width
        self.y = new_y % self.env_height

    def rotate(self):
        self.angle += self.rotate_deg
        self.angle %= 360  # On garde l'angle entre 0 et 360.

    def eat(self, victim):
        victim.alive = False
        self.energy += victim.energy
        victim.energy = 0

    def digestion(self):
        waste = self.stomach
        self.stomach = 0
        return waste

    def division(self, id_agent, vision_angle, pd, pi, wmax, iteration):
        # On calcule un angle sûr dans la zone aveugle du parent
        angle_enfant = (self.angle + random.uniform(vision_angle,
                                                    360 - vision_angle)) % 360  # Modulo 360, car on veut garder l'angle entre 0 et 360
        angle_enfant_rad = math.radians(angle_enfant)

        # On utilise cet angle pour placer l'enfant à SPAWN_DISTANCE pixels dans une
        # direction différente de la vision du parent (modulation sur le tore).
        new_x = (self.x + self.SPAWN_DISTANCE * math.cos(angle_enfant_rad)) % self.env_width
        new_y = (self.y + self.SPAWN_DISTANCE * math.sin(angle_enfant_rad)) % self.env_height

        # On crée un nouvel agent enfant
        enfant = Agent(id_agent, new_x, new_y, self.energy, self.rotate_deg, self.env_width, self.env_height,
                       self.cost_rotate, self.cost_move, self.cost_eat, self.cost_neuron, self.cost_metabolism,
                       self.max_cout_A, self.max_cout_B)
        enfant.parent = self  # Lier l'enfant au parent
        # On copie le cerveau du parent dans l'enfant (copie profonde complète)
        enfant.sbn = copy.deepcopy(self.sbn)

        # On divise par deux l'énergie de l'enfant et du parent
        enfant.energy /= 2
        self.energy /= 2

        # On donne à l'enfant son angle de vision
        enfant.angle = angle_enfant

        # On fait évoluer l'enfant à la naissance
        enfant.sbn.mutation(pd, pi, wmax)

        return enfant

    def update(self, oeil, pw, wmax):  # pd et pi ne servent pas, car l'agent ne voit pas ses neoud changé au cours de sa vie.
        # Si l'agent est mort, on ne fait rien
        if not self.alive:
            return None

        self.energy -= self.cost_metabolism
        lost_energy = self.cost_metabolism
        # On appelle la mutation
        # self.sbn.mutation(pd, pi, wmax)
        # self.sbn.evolution(pw, wmax)

        # On récupère les actions ordonnées par le cerveau
        action_eat, action_move, action_rotate = self.sbn.step(oeil)

        # On retire pour chaque neurone activé son coût
        # value_cost_neuron = float(np.sum(self.sbn.states)) * self.cost_neuron
        value_cost_neuron = float(np.sum(self.sbn.states)) * self.calculer_contrainte()  # ✅ Remplace self.cost_neuron
        lost_energy += value_cost_neuron
        self.energy -= value_cost_neuron

        # On effectue chaque action en vérifiant l'énergie et en la diminuant en fonction
        if action_rotate and self.energy >= self.cost_rotate:
            self.rotate()
            self.energy -= self.cost_rotate
            lost_energy += self.cost_rotate
        if action_move and self.energy >= self.cost_move:
            self.move()
            self.energy -= self.cost_move
            lost_energy += self.cost_move
        if action_eat:
            if self.energy >= self.cost_eat:
                self.energy -= self.cost_eat
                lost_energy += self.cost_eat
            else:
                action_eat = 0

        # Si l'agent n'a plus assez d'énergie, il est donc mort
        if self.energy <= self.DEATH_ENERGY:
            self.alive = False

        # On incrémente le compteur de pas
        self.step += 1

        # On incrémente l'estomac avec l'énergie dépensée
        self.stomach += lost_energy

        # On retourne l'action manger pour que l'environnement puisse le savoir
        # (un agent mort ce pas-ci ne peut plus manger)
        return 0 if not self.alive else action_eat

    def sense(self, voisins, dist_vision_sq, dist_manger_sq, angle_vision):
        self.vision_input = 0
        self.proie_potentielle = None
        angle_regard = math.radians(self.angle)
        angle_vision_rad = math.radians(angle_vision)

        for other in voisins:
            if other is self or not other.alive:
                continue

            # On calcule la distance entre les deux agents : vecteur le plus court
            # sur le tore (l'environnement est enroulé, cf. move()).
            dx = other.x - self.x
            if dx > self.env_width / 2:
                dx -= self.env_width
            elif dx < -self.env_width / 2:
                dx += self.env_width
            dy = other.y - self.y
            if dy > self.env_height / 2:
                dy -= self.env_height
            elif dy < -self.env_height / 2:
                dy += self.env_height
            dist_sq = dx * dx + dy * dy

            # On regarde si dans les voisins de l'agent, il y en a un dans son cone de vision
            if dist_sq < dist_vision_sq:
                angle_cible = math.atan2(dy, dx)
                diff = (angle_cible - angle_regard + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) < angle_vision_rad:
                    self.vision_input = 1
                    # On vérifie qu'il est à bonne distance
                    if dist_sq < dist_manger_sq:
                        self.proie_potentielle = other
                        break  # On arrête à la première proie vue et mangeable

    def draw(self, screen, overlay, size, vision_dist, fov, max_energy, offset_y, vision_cone, tracking=False):
        if not self.alive:
            return

        # On récupère la position de l'agent
        pos = (int(self.x),
               int(self.y + offset_y))  # offset crée le décalage pour laisser de la place pour le dashboard du haut

        # Affichage du cercle coloré
        ratio = max(0, min(self.energy / max_energy, 1))
        color = (int(255 * (1 - ratio)), int(255 * ratio), 0)
        if tracking:  # Si l'agent est ciblé
            # Halo de ciblage (cercle vide plus grand)
            pygame.draw.circle(screen, (255, 255, 255), pos, size + 10, 2)

        pygame.draw.circle(screen, color, pos, size)

        if vision_cone:
            # Affichage du cône de vision
            left_rad = math.radians(self.angle - fov)
            right_rad = math.radians(self.angle + fov)

            p_left = (self.x + vision_dist * math.cos(left_rad), (self.y + offset_y) + vision_dist * math.sin(left_rad))
            p_right = (self.x + vision_dist * math.cos(right_rad),
                       (self.y + offset_y) + vision_dist * math.sin(right_rad))

            cone_color = (255, 0, 0, 40) if self.vision_input == 1 else (255, 255, 255, 20)
            pygame.draw.polygon(overlay, cone_color, [pos, p_left, p_right])

    def calculer_contrainte(self):
        gradient_A = (self.x / self.env_width) * self.max_cout_A
        gradient_B = (self.y / self.env_height) * self.max_cout_B

        en_A = self.x > self.env_width / 2
        en_B = self.y > self.env_height / 2

        if en_A and en_B:
            return self.cout_base + 0.5 * self.max_cout_A + 0.5 * self.max_cout_B
        elif en_A:
            return self.cout_base + gradient_A
        elif en_B:
            return self.cout_base + gradient_B
        else:
            return self.cout_base