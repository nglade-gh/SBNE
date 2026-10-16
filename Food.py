import pygame
class Food:
    def __init__(self, x, y, energy=50, active=True):
        self.x = x
        self.y = y
        self.energy = energy
        self.alive = True
        self.active = active
        
    def draw(self, screen, offset_y):
        
        if self.alive:
            if self.active:
                color = (255, 0, 255)
            else:
                color = (137, 81, 41)
            # Cercle de couleur jaune pour la nourriture
            pygame.draw.circle(screen, color, (int(self.x), int(self.y + offset_y)), 4)