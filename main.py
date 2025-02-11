import pygame
import numpy as np
import random
import math
import time

# -------------------------------
# Global Constants and Terrain
# -------------------------------
WIDTH, HEIGHT = 800, 600
FPS = 30
DAY_LENGTH = 600            # Frames per day cycle
PLANT_REGROW_TIME = 300     # Frames until a plant regrows
WATER_REGROW_TIME = 400     # Frames until a water source regrows
BOAR_FORAGE_DURATION = 100  # Frames needed for a boar to complete foraging

# Define some terrain zones (which slow movement)
TERRAIN_ZONES = [
    (pygame.Rect(100, 100, 200, 200), 0.8),  # Forest
    (pygame.Rect(500, 50, 150, 150), 0.5)      # Water
]

def get_terrain_factor(pos):
    factor = 1.0
    for rect, mod in TERRAIN_ZONES:
        if rect.collidepoint(pos[0], pos[1]):
            factor = min(factor, mod)
    return factor

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# -------------------------------
# Resource Classes
# -------------------------------
class Plant:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.active = True
        self.regrow_timer = 0
    def update(self):
        if not self.active:
            self.regrow_timer -= 1
            if self.regrow_timer <= 0:
                self.active = True

class WaterSource:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.active = True
        self.regrow_timer = 0
    def update(self):
        if not self.active:
            self.regrow_timer -= 1
            if self.regrow_timer <= 0:
                self.active = True

# -------------------------------
# Base Creature Class
# -------------------------------
class Creature:
    def __init__(self, x, y, health, base_speed):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0], dtype=float)  # Ensure vel is float64
        self.health = health
        self.max_health = health
        self.base_speed = base_speed
    def update_position(self):
        self.pos += self.vel
        self.pos[0] = max(0, min(WIDTH, self.pos[0]))
        self.pos[1] = max(0, min(HEIGHT, self.pos[1]))

# -------------------------------
# Goblin Class (Q–Learning, Grouping, Overrides to Attack Predators)
# All goblins are now red.
# -------------------------------
class Goblin(Creature):
    def __init__(self, x, y,
                 q_weights1=None, q_bias1=None, q_weights2=None, q_bias2=None,
                 alpha=0.01, gamma=0.9, epsilon=0.1):
        # Improved skills: higher speed (3.0) and higher attack damage (15)
        super().__init__(x, y, 50, base_speed=3.0)
        self.energy = 0
        self.attack_damage = 15
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        # Q–network: input dim 2, hidden 4, output 2 (0: hunt boars, 1: forage)
        self.q_weights1 = q_weights1 if q_weights1 is not None else np.random.randn(2, 4)
        self.q_bias1    = q_bias1 if q_bias1 is not None else np.random.randn(4)
        self.q_weights2 = q_weights2 if q_weights2 is not None else np.random.randn(4, 2)
        self.q_bias2    = q_bias2 if q_bias2 is not None else np.random.randn(2)
        self.exp_buffer = []
        self.buffer_size = 20
        self.batch_size = 4
        self.last_state = np.zeros(2)
        self.last_action = 0
        # All goblins are red
        self.color = (255, 0, 0)
    def get_state(self, boars, plants):
        boar_near = 1.0 if any(distance(self.pos, b.pos) < 150 for b in boars) else 0.0
        plant_near = 1.0 if any(p.active and distance(self.pos, p.pos) < 150 for p in plants) else 0.0
        return np.array([boar_near, plant_near])
    def q_forward(self, state):
        hidden = np.tanh(np.dot(state, self.q_weights1) + self.q_bias1)
        return np.dot(hidden, self.q_weights2) + self.q_bias2
    def choose_action(self, boars, plants):
        state = self.get_state(boars, plants)
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            q_vals = self.q_forward(state)
            action = int(np.argmax(q_vals))
        self.last_state = state
        self.last_action = action
        return action
    def store_experience(self, reward, boars, plants):
        next_state = self.get_state(boars, plants)
        self.exp_buffer.append((self.last_state, self.last_action, reward, next_state))
        if len(self.exp_buffer) > self.buffer_size:
            self.exp_buffer.pop(0)
    def train_q_network(self):
        if len(self.exp_buffer) < self.batch_size:
            return
        batch = random.sample(self.exp_buffer, self.batch_size)
        grad_w1 = np.zeros_like(self.q_weights1)
        grad_b1 = np.zeros_like(self.q_bias1)
        grad_w2 = np.zeros_like(self.q_weights2)
        grad_b2 = np.zeros_like(self.q_bias2)
        for state, action, reward, next_state in batch:
            hidden = np.tanh(np.dot(state, self.q_weights1) + self.q_bias1)
            q_vals = np.dot(hidden, self.q_weights2) + self.q_bias2
            target = reward + self.gamma * np.max(self.q_forward(next_state))
            error = target - q_vals[action]
            dloss_dq = -2 * error
            grad_w2 += np.outer(hidden, np.eye(2)[action] * dloss_dq)
            grad_b2 += np.eye(2)[action] * dloss_dq
            dtanh = 1 - hidden**2
            dhidden = (dloss_dq * self.q_weights2[:, action]) * dtanh
            grad_w1 += np.outer(state, dhidden)
            grad_b1 += dhidden
        self.q_weights1 -= self.alpha * (grad_w1 / self.batch_size)
        self.q_bias1    -= self.alpha * (grad_b1 / self.batch_size)
        self.q_weights2 -= self.alpha * (grad_w2 / self.batch_size)
        self.q_bias2    -= self.alpha * (grad_b2 / self.batch_size)
    def update(self, boars, goblins, plants, predators, speed_global=1.0):
        # If any predator is within 100 pixels, override to target predator.
        predator_target = None
        min_pd = float('inf')
        for p in predators:
            d = distance(self.pos, p.pos)
            if d < min_pd:
                min_pd = d
                predator_target = p
        if predator_target is not None and min_pd < 100:
            direction = (predator_target.pos - self.pos)
            norm = np.linalg.norm(direction) + 1e-5
            self.vel = (direction / norm) * self.base_speed * speed_global
        else:
            action = self.choose_action(boars, plants)  # 0: hunt boars, 1: forage
            if action == 0:
                closest = None
                min_d = float('inf')
                for b in boars:
                    d = distance(self.pos, b.pos)
                    if d < min_d:
                        min_d = d
                        closest = b
                if closest:
                    direction = (closest.pos - self.pos)
                    norm = np.linalg.norm(direction) + 1e-5
                    move = direction / norm
                else:
                    move = np.array([0, 0])
                self.vel = move * self.base_speed * speed_global
            else:
                closest = None
                min_d = float('inf')
                for p in plants:
                    if p.active:
                        d = distance(self.pos, p.pos)
                        if d < min_d:
                            min_d = d
                            closest = p
                if closest:
                    direction = (closest.pos - self.pos)
                    norm = np.linalg.norm(direction) + 1e-5
                    move = direction / norm
                else:
                    move = np.array([0, 0])
                self.vel = move * self.base_speed * speed_global
        # Separation force: avoid overlapping with other goblins
        separation = np.array([0.0, 0.0])
        count = 0
        for other in goblins:
            if other is not self:
                d = distance(self.pos, other.pos)
                if d < 20:
                    separation += (self.pos - other.pos) / (d + 1e-5)
                    count += 1
        if count > 0:
            separation = separation / count
            self.vel += separation * 0.5
        # Repulsion from nearby predators
        predator_repulse = np.array([0.0, 0.0])
        count_pred = 0
        for p in predators:
            d = distance(self.pos, p.pos)
            if d < 30:
                predator_repulse += (self.pos - p.pos) / (d + 1e-5)
                count_pred += 1
        if count_pred > 0:
            predator_repulse = predator_repulse / count_pred
            self.vel += predator_repulse * 1.0
        self.vel *= get_terrain_factor(self.pos)
        self.update_position()
        self.train_q_network()
    def try_attack(self, boars, predators, plants):
        # Attack boars if close...
        for b in boars[:]:
            if distance(self.pos, b.pos) < 10:
                b.health -= self.attack_damage
                if b.health <= 0:
                    self.energy += 50
                    self.store_experience(50, boars, plants)
                    boars.remove(b)
        # Also attack predators if nearby
        for p in predators[:]:
            if distance(self.pos, p.pos) < 10:
                p.health -= self.attack_damage
                if p.health <= 0:
                    self.energy += 50
                    self.store_experience(50, boars, plants)
                    predators.remove(p)
    def try_forage(self, plants, boars):
        for p in plants:
            if p.active and distance(self.pos, p.pos) < 10:
                self.energy += 20
                self.store_experience(20, boars, plants)
                p.active = False
                p.regrow_timer = PLANT_REGROW_TIME
                break
    def try_reproduce(self, goblin_list):
        threshold = 60  # Lower reproduction threshold for goblins
        if self.energy >= threshold:
            for other in goblin_list:
                if other is not self and distance(self.pos, other.pos) < 20:
                    new_x = (self.pos[0] + other.pos[0]) / 2 + random.uniform(-5, 5)
                    new_y = (self.pos[1] + other.pos[1]) / 2 + random.uniform(-5, 5)
                    new_base_speed = max(2.0, self.base_speed + random.gauss(0, 0.1))
                    new_attack_damage = max(5, self.attack_damage + random.gauss(0, 0.5))
                    new_q_w1 = self.q_weights1 + np.random.randn(*self.q_weights1.shape) * 0.01
                    new_q_b1 = self.q_bias1 + np.random.randn(*self.q_bias1.shape) * 0.01
                    new_q_w2 = self.q_weights2 + np.random.randn(*self.q_weights2.shape) * 0.01
                    new_q_b2 = self.q_bias2 + np.random.randn(*self.q_bias2.shape) * 0.01
                    new_epsilon = max(0.01, self.epsilon + random.gauss(0, 0.005))
                    child = Goblin(new_x, new_y,
                                   q_weights1=new_q_w1, q_bias1=new_q_b1,
                                   q_weights2=new_q_w2, q_bias2=new_q_b2,
                                   alpha=self.alpha, gamma=self.gamma, epsilon=new_epsilon)
                    child.base_speed = new_base_speed
                    child.attack_damage = new_attack_damage
                    self.energy -= threshold / 2
                    other.energy -= threshold / 2
                    goblin_list.append(child)
                    break

# -------------------------------
# Boar Class (Beast)
# -------------------------------
class Boar(Creature):
    def __init__(self, x, y, species=None, action_weights=None):
        super().__init__(x, y, 100, base_speed=1.5)
        self.vel = np.array([0.0, 0.0], dtype=np.float64)  # Ensure vel is float64
        self.foraging_counter = 0
        self.reproduction_weight = 1.0
        self.state = "wandering"
        self.species = species if species else random.choice(["boarA", "boarB"])
        self.action_weights = action_weights if action_weights else {"forage": 1.0, "wander": 1.0, "flee": 1.0}
        self.attack_damage = 5
    def update(self, goblins, plants, speed_global=1.0):
        plant_close = None
        for p in plants:
            if p.active and distance(self.pos, p.pos) < 15:
                plant_close = p
                break
        if plant_close:
            total = self.action_weights["forage"] + self.action_weights["wander"]
            p_forage = self.action_weights["forage"] / total
            if random.random() < p_forage:
                self.state = "foraging"
                self.foraging_counter += 1
                self.vel = np.array([0, 0])
            else:
                self.state = "wandering"
                rand_dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                norm = np.linalg.norm(rand_dir) + 1e-5
                self.vel = (rand_dir / norm) * self.base_speed * speed_global
                self.foraging_counter = max(0, self.foraging_counter - 1)
        else:
            threats = [g.pos for g in goblins if distance(self.pos, g.pos) < 100]
            if threats:
                self.state = "fleeing"
                avg = np.mean(threats, axis=0)
                direction = self.pos - avg
                norm = np.linalg.norm(direction) + 1e-5
                self.vel = (direction / norm) * 3 * speed_global
                self.foraging_counter = 0
            else:
                self.state = "wandering"
                rand_dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                norm = np.linalg.norm(rand_dir) + 1e-5
                self.vel = (rand_dir / norm) * self.base_speed * speed_global
                self.foraging_counter = max(0, self.foraging_counter - 1)
        self.vel = self.vel.astype(np.float64)  # Ensure vel is float64
        self.vel *= np.float64(get_terrain_factor(self.pos))
        self.update_position()

    
    def try_attack(self, goblins):
        for g in goblins[:]:
            if distance(self.pos, g.pos) < 10:
                g.health -= self.attack_damage
                if g.health <= 0:
                    self.reproduction_weight += 0.5
    def try_reproduce(self, boar_list):
        if (self.foraging_counter >= BOAR_FORAGE_DURATION or self.reproduction_weight >= 2.0):
            for other in boar_list:
                if other is not self and other.species == self.species:
                    if (other.foraging_counter >= BOAR_FORAGE_DURATION or other.reproduction_weight >= 2.0) and distance(self.pos, other.pos) < 20:
                        new_x = (self.pos[0] + other.pos[0]) / 2 + random.uniform(-5, 5)
                        new_y = (self.pos[1] + other.pos[1]) / 2 + random.uniform(-5, 5)
                        new_weights = {k: self.action_weights[k] + random.gauss(0, 0.01) for k in self.action_weights}
                        child = Boar(new_x, new_y, species=self.species, action_weights=new_weights)
                        self.foraging_counter = 0
                        other.foraging_counter = 0
                        self.reproduction_weight = 1.0
                        other.reproduction_weight = 1.0
                        boar_list.append(child)
                        break

# -------------------------------
# Scavenger Class (Giant Spiders - Very Solitary)
# -------------------------------
class Scavenger(Creature):
    def __init__(self, x, y, species=None, action_weights=None):
        super().__init__(x, y, 40, base_speed=2.5)
        self.energy = 0
        self.steal_amount = 20
        self.species = species if species else random.choice(["spiderA", "spiderB"])
        self.action_weights = action_weights if action_weights else {"steal": 1.0, "forage": 1.0}
        self.attack_damage = 3
        self.weights1 = np.random.randn(2, 4)
        self.bias1 = np.random.randn(4)
        self.weights2 = np.random.randn(4, 2)
        self.bias2 = np.random.randn(2)
    def decide_action(self, goblins, plants):
        targets = [g for g in goblins if g.energy > 20 and distance(self.pos, g.pos) < 150]
        if targets:
            tot = self.action_weights["steal"] + self.action_weights["forage"]
            if random.random() < (self.action_weights["steal"] / tot):
                return "steal"
        return "forage"
    def decide_movement_steal(self, goblins):
        target = None
        min_d = float('inf')
        for g in goblins:
            if g.energy > 20:
                d = distance(self.pos, g.pos)
                if d < min_d:
                    min_d = d
                    target = g
        if target:
            direction = target.pos - self.pos
            norm = np.linalg.norm(direction) + 1e-5
            return direction / norm
        return np.array([0, 0])
    def decide_movement_forage(self, plants):
        closest = None
        min_d = float('inf')
        for p in plants:
            if p.active:
                d = distance(self.pos, p.pos)
                if d < min_d:
                    min_d = d
                    closest = p
        if closest:
            direction = closest.pos - self.pos
            norm = np.linalg.norm(direction) + 1e-5
            return direction / norm
        return np.array([0, 0])
    def update(self, goblins, plants, scavengers, speed_global=1.0):
        action = self.decide_action(goblins, plants)
        if action == "steal":
            move_direction = self.decide_movement_steal(goblins)
            hidden = np.tanh(np.dot(move_direction, self.weights1) + self.bias1)
            output = np.tanh(np.dot(hidden, self.weights2) + self.bias2)
            self.vel = output * self.base_speed * speed_global
        else:
            move_direction = self.decide_movement_forage(plants)
            self.vel = move_direction * self.base_speed * speed_global
        # --- Separation: stay far from other scavengers
        separation = np.array([0.0, 0.0])
        count = 0
        for other in scavengers:
            if other is not self:
                d = distance(self.pos, other.pos)
                if d < 40:
                    separation += (self.pos - other.pos) / (d + 1e-5)
                    count += 1
        if count > 0:
            separation = separation / count
            self.vel += separation * 2.0  # Strong repulsion
        self.vel *= get_terrain_factor(self.pos)
        self.update_position()
    def try_steal(self, goblins):
        for g in goblins:
            if g.energy > 0 and distance(self.pos, g.pos) < 10:
                stolen = min(self.steal_amount, g.energy)
                g.energy -= stolen
                self.energy += stolen
                self.action_weights["steal"] *= 1.1
                break
    def try_forage(self, plants):
        for p in plants:
            if p.active and distance(self.pos, p.pos) < 10:
                self.energy += 20
                self.action_weights["forage"] *= 1.05
                p.active = False
                p.regrow_timer = PLANT_REGROW_TIME
                break
    def try_reproduce(self, scavenger_list):
        threshold = 120  # Higher threshold for scavengers
        if self.energy >= threshold:
            for other in scavenger_list:
                if other is not self and other.species == self.species and other.energy >= threshold:
                    if distance(self.pos, other.pos) < 20:
                        new_x = (self.pos[0] + other.pos[0]) / 2 + random.uniform(-5, 5)
                        new_y = (self.pos[1] + other.pos[1]) / 2 + random.uniform(-5, 5)
                        new_aw = {k: self.action_weights[k] + random.gauss(0, 0.01) for k in self.action_weights}
                        child = Scavenger(new_x, new_y, species=self.species, action_weights=new_aw)
                        self.energy -= threshold / 2
                        other.energy -= threshold / 2
                        scavenger_list.append(child)
                        break

# -------------------------------
# Predator Class (Wolves) – Now also target scavengers
# Also predators maintain solitary spacing.
# -------------------------------
class Predator(Creature):
    def __init__(self, x, y, species=None):
        super().__init__(x, y, 80, base_speed=3.0)
        self.energy = 0
        self.species = species if species else random.choice(["wolfA", "wolfB"])
        self.attack_damage = 10  # Reduced damage from before
    def update(self, goblins, boars, scavengers, predators, speed_global=1.0):
        # Chase nearest prey among goblins, boars, and scavengers
        prey_list = goblins + boars + scavengers
        target = None
        min_d = float('inf')
        for p in prey_list:
            d = distance(self.pos, p.pos)
            if d < min_d:
                min_d = d
                target = p
        if target:
            direction = target.pos - self.pos
            norm = np.linalg.norm(direction) + 1e-5
            self.vel = (direction / norm) * self.base_speed * speed_global
        else:
            rand_dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
            norm = np.linalg.norm(rand_dir) + 1e-5
            self.vel = (rand_dir / norm) * self.base_speed * speed_global
        # Separation: predators stay solitary
        repulsion = np.array([0.0, 0.0])
        count = 0
        for other in predators:
            if other is not self:
                d = distance(self.pos, other.pos)
                if d < 50:
                    repulsion += (self.pos - other.pos) / (d + 1e-5)
                    count += 1
        if count > 0:
            repulsion = repulsion / count
            self.vel += repulsion * 1.0
        self.vel *= get_terrain_factor(self.pos)
        self.update_position()
    def try_attack(self, goblins, boars, scavengers):
        prey_list = goblins + boars + scavengers
        for p in prey_list[:]:
            if distance(self.pos, p.pos) < 10:
                p.health -= self.attack_damage
                if p.health <= 0:
                    self.energy += 50
                    if p in goblins:
                        goblins.remove(p)
                    elif p in boars:
                        boars.remove(p)
                    elif p in scavengers:
                        scavengers.remove(p)
    def try_reproduce(self, predator_list):
        threshold = 200
        if self.energy >= threshold:
            for other in predator_list:
                if other is not self and other.species == self.species and other.energy >= threshold:
                    if distance(self.pos, other.pos) < 20:
                        new_x = (self.pos[0] + other.pos[0]) / 2 + random.uniform(-5, 5)
                        new_y = (self.pos[1] + other.pos[1]) / 2 + random.uniform(-5, 5)
                        child = Predator(new_x, new_y, species=self.species)
                        self.energy -= threshold / 2
                        other.energy -= threshold / 2
                        predator_list.append(child)
                        break

# -------------------------------
# Main Simulation Loop, UI, Logging, and External Spawning
# -------------------------------
def main():
    pygame.init()
    try:
        pygame.mixer.init()
    except Exception as e:
        print("Audio initialization failed:", e)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ecosystem Simulator – Revised Edition")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    
    try:
        attack_sound = pygame.mixer.Sound("attack.wav")
        reproduce_sound = pygame.mixer.Sound("reproduce.wav")
    except:
        attack_sound = None
        reproduce_sound = None

    log_file = open("simulation_log.txt", "a")
    
    # Increase initial goblins; use 20 instead of 8.
    goblins = [Goblin(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(20)]
    boars = [Boar(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(5)]
    scavengers = [Scavenger(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(3)]
    predators = [Predator(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(2)]
    plants = [Plant(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(30)]
    water_sources = [WaterSource(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(5)]
    
    frame_count = 0
    
    running = True
    while running:
        clock.tick(FPS)
        frame_count += 1
        speed_factor = 1.0  # Constant speed factor
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        for plant in plants:
            plant.update()
        for water in water_sources:
            water.update()
        if len(plants) < 35 and random.random() < 0.02:
            plants.append(Plant(random.randint(0, WIDTH), random.randint(0, HEIGHT)))
        if len(water_sources) < 8 and random.random() < 0.01:
            water_sources.append(WaterSource(random.randint(0, WIDTH), random.randint(0, HEIGHT)))
        
        # Update creatures
        for g in goblins[:]:
            g.update(boars, goblins, plants, predators, speed_global=speed_factor)
            g.try_attack(boars, predators, plants)
            g.try_forage(plants, boars)
            g.try_reproduce(goblins)
            if g.health <= 0:
                goblins.remove(g)
        for b in boars[:]:
            b.update(goblins, plants, speed_global=speed_factor)
            b.try_attack(goblins)
            b.try_reproduce(boars)
            if b.health <= 0:
                boars.remove(b)
        for s in scavengers[:]:
            s.update(goblins, plants, scavengers, speed_global=speed_factor)
            s.try_steal(goblins)
            s.try_forage(plants)
            s.try_reproduce(scavengers)
            if s.health <= 0:
                scavengers.remove(s)
        for p in predators[:]:
            p.update(goblins, boars, scavengers, predators, speed_global=speed_factor)
            p.try_attack(goblins, boars, scavengers)
            p.try_reproduce(predators)
            if p.health <= 0:
                predators.remove(p)
        
        # External spawning only if population of that type is less than 3.
        if len(boars) < 3 and random.random() < 0.005:
            edge = random.randint(0, 3)
            if edge == 0:
                x, y = -10, random.randint(0, HEIGHT)
            elif edge == 1:
                x, y = WIDTH + 10, random.randint(0, HEIGHT)
            elif edge == 2:
                x, y = random.randint(0, WIDTH), -10
            else:
                x, y = random.randint(0, WIDTH), HEIGHT + 10
            boars.append(Boar(x, y))
        if len(scavengers) < 3 and random.random() < 0.005:
            edge = random.randint(0, 3)
            if edge == 0:
                x, y = -10, random.randint(0, HEIGHT)
            elif edge == 1:
                x, y = WIDTH + 10, random.randint(0, HEIGHT)
            elif edge == 2:
                x, y = random.randint(0, WIDTH), -10
            else:
                x, y = random.randint(0, WIDTH), HEIGHT + 10
            scavengers.append(Scavenger(x, y))
        if len(predators) < 3 and random.random() < 0.003:
            edge = random.randint(0, 3)
            if edge == 0:
                x, y = -10, random.randint(0, HEIGHT)
            elif edge == 1:
                x, y = WIDTH + 10, random.randint(0, HEIGHT)
            elif edge == 2:
                x, y = random.randint(0, WIDTH), -10
            else:
                x, y = random.randint(0, WIDTH), HEIGHT + 10
            predators.append(Predator(x, y))
        
        bg_color = (50, 50, 80)  # Constant background color
        screen.fill(bg_color)
        
        for rect, mod in TERRAIN_ZONES:
            pygame.draw.rect(screen, (30, 60, 30), rect, 2)
        for water in water_sources:
            if water.active:
                pygame.draw.circle(screen, (0, 191, 255), (int(water.pos[0]), int(water.pos[1])), 6)
        for plant in plants:
            if plant.active:
                pygame.draw.circle(screen, (0,255,0), (int(plant.pos[0]), int(plant.pos[1])), 5)
        for g in goblins:
            pygame.draw.circle(screen, g.color, (int(g.pos[0]), int(g.pos[1])), 5)
            end = (int(g.pos[0] + g.vel[0]*5), int(g.pos[1] + g.vel[1]*5))
            pygame.draw.line(screen, (0,0,0), (int(g.pos[0]), int(g.pos[1])), end, 2)
        for b in boars:
            col = (139,69,19) if b.state != "foraging" else (0,0,255)
            pygame.draw.circle(screen, col, (int(b.pos[0]), int(b.pos[1])), 8)
        for s in scavengers:
            pygame.draw.circle(screen, (128,0,128), (int(s.pos[0]), int(s.pos[1])), 6)
        for p in predators:
            pygame.draw.circle(screen, (100,100,100), (int(p.pos[0]), int(p.pos[1])), 7)
        
        metrics = (f"Goblins: {len(goblins)}  Boars: {len(boars)}  Scavengers: {len(scavengers)}  "
                   f"Predators: {len(predators)}  Plants: {len(plants)}  Water: {len(water_sources)}")
        text_surface = font.render(metrics, True, (255,255,255))
        screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
        
        if frame_count % 60 == 0:
            log_line = f"{time.time()},{len(goblins)},{len(boars)},{len(scavengers)},{len(predators)}\n"
            log_file.write(log_line)
            log_file.flush()
    
    log_file.close()
    pygame.quit()

if __name__ == "__main__":
    main()