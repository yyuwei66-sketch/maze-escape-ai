"""
demo_pygame.py
==============
Standalone, runnable demo of the monster AI.  It is also the reference for
how the teammate's main project should call the module: the ONLY coupling
is the single `controller.decide(...)` call marked `INTEGRATION POINT`.

Run:
    pip install pygame
    python demo_pygame.py

Controls:
    Arrow keys / WASD : move the human one cell (board wraps around)
    Space             : wait one turn (don't move)
    1                 : greedy level
    2                 : minimax level (harder)
    R                 : new round / respawn
    Esc               : quit
"""

import random
import sys

import pygame

from monster_ai import make_monster_controller, TorusGrid

# ---- configuration ---------------------------------------------------
GRID = 30
CELL = 22
MARGIN = 16
HUD = 56
STEPS_PER_TURN = 2
WALL_DENSITY = 0.10
MINIMAX_DEPTH = 2

W = GRID * CELL + 2 * MARGIN
H = GRID * CELL + 2 * MARGIN + HUD

# placeholder "art" — flat colours; the artist replaces these with sprites
C_BG = (24, 26, 34)
C_GRID = (38, 41, 54)
C_WALL = (70, 78, 102)
C_PLAYER = (90, 200, 160)
C_MON = [(232, 96, 96), (240, 168, 72)]
C_TEXT = (220, 224, 235)
C_TRAIL = (52, 58, 78)


def random_walls(seed):
    rng = random.Random(seed)
    return {(rng.randrange(GRID), rng.randrange(GRID))
            for _ in range(int(GRID * GRID * WALL_DENSITY))}


class Game:
    def __init__(self, algorithm="minimax", seed=0):
        self.seed = seed
        self.set_level(algorithm)

    def set_level(self, algorithm):
        self.algorithm = algorithm
        self.walls = random_walls(self.seed)
        self.grid = TorusGrid(GRID, GRID, self.walls)
        self.controller = make_monster_controller(
            algorithm, self.walls, GRID, GRID,
            steps_per_turn=STEPS_PER_TURN, depth=MINIMAX_DEPTH,
        )
        self.reset()

    def reset(self):
        free = [self.grid.coord(i) for i in range(self.grid.n)
                if not self.grid.blocked[i]]
        rng = random.Random(self.seed * 7 + 13)
        self.player = rng.choice(free)
        self.monsters = []
        while len(self.monsters) < 2:
            c = rng.choice(free)
            if (c != self.player and c not in self.monsters
                    and self.grid.toroidal_manhattan(c, self.player) > 8):
                self.monsters.append(c)
        self.trails = []          # cells the monsters stepped through (for fx)
        self.over = False
        self.win_turns = 0

    # ---- one full turn -----------------------------------------------
    def step(self, dx, dy):
        if self.over:
            return
        # 1) player moves one cell (wrap-around)
        self.player = ((self.player[0] + dx) % GRID,
                       (self.player[1] + dy) % GRID)
        if self.player in self.walls:            # bumped a wall -> undo
            self.player = ((self.player[0] - dx) % GRID,
                           (self.player[1] - dy) % GRID)
        self.win_turns += 1
        if self.player in self.monsters:
            self.over = True
            return

        # 2) ===== INTEGRATION POINT ===================================
        #    Hand the AI the post-move player position and the monster
        #    positions; get back one route per monster for this turn.
        paths = self.controller.decide(self.player, self.monsters)
        # =============================================================
        self.trails = [c for pth in paths for c in pth[1:]]
        self.monsters = [pth[-1] for pth in paths]

        # capture if a monster passed THROUGH the player on either step
        if any(self.player in pth for pth in paths):
            self.over = True


def draw(screen, font, game):
    screen.fill(C_BG)
    ox = oy = MARGIN

    def rect(cx, cy):
        return pygame.Rect(ox + cx * CELL, oy + cy * CELL, CELL - 1, CELL - 1)

    for y in range(GRID):
        for x in range(GRID):
            pygame.draw.rect(screen, C_GRID, rect(x, y), 1)
    for (x, y) in game.walls:
        pygame.draw.rect(screen, C_WALL, rect(x, y))
    for (x, y) in game.trails:
        pygame.draw.rect(screen, C_TRAIL, rect(x, y))
    px, py = game.player
    pygame.draw.rect(screen, C_PLAYER, rect(px, py), border_radius=6)
    for i, (mx, my) in enumerate(game.monsters):
        pygame.draw.rect(screen, C_MON[i % len(C_MON)], rect(mx, my),
                         border_radius=10)

    hud_y = MARGIN + GRID * CELL + 8
    msg = (f"LEVEL: {game.algorithm.upper()}   turns: {game.win_turns}"
           "   [1]greedy [2]minimax [R]new [Space]wait")
    screen.blit(font.render(msg, True, C_TEXT), (MARGIN, hud_y))
    if game.over:
        big = font.render("CAUGHT!  press R", True, (255, 120, 120))
        screen.blit(big, (MARGIN, hud_y + 22))
    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Monster AI demo  (greedy / minimax)")
    font = pygame.font.Font(None, 20)
    game = Game(algorithm="minimax", seed=0)
    clock = pygame.time.Clock()

    MOVES = {
        pygame.K_UP: (0, -1), pygame.K_w: (0, -1),
        pygame.K_DOWN: (0, 1), pygame.K_s: (0, 1),
        pygame.K_LEFT: (-1, 0), pygame.K_a: (-1, 0),
        pygame.K_RIGHT: (1, 0), pygame.K_d: (1, 0),
        pygame.K_SPACE: (0, 0),
    }

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                elif ev.key == pygame.K_1:
                    game.set_level("greedy")
                elif ev.key == pygame.K_2:
                    game.set_level("minimax")
                elif ev.key == pygame.K_r:
                    game.reset()
                elif ev.key in MOVES:
                    game.step(*MOVES[ev.key])
        draw(screen, font, game)
        clock.tick(60)


if __name__ == "__main__":
    main()

