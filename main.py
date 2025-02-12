import pygame
import numpy as np
import random
import math
import time
import os
import json
from dataclasses import dataclass
from typing import List

# -------------------------------
# Global Constants and Terrain
# -------------------------------
WIDTH, HEIGHT = 1280, 720
FPS = 30
DAY_LENGTH = 600            # Frames per day cycle
PLANT_REGROW_TIME = 300     # Frames until a plant regrows
WATER_REGROW_TIME = 400     # Frames until a water source regrows
BOAR_FORAGE_DURATION = 100  # Frames needed for a boar to complete foraging

@dataclass
class TerrainZone:
    """Class representing a terrain zone with a slowdown modifier."""
    rect: pygame.Rect
    modifier: float

# List of terrain zones where movement is slowed
TERRAIN_ZONES: List[TerrainZone] = [
    TerrainZone(pygame.Rect(100, 100, 200, 200), 0.8),  # Forest
    TerrainZone(pygame.Rect(500, 50, 150, 150), 0.5)      # Water
]

def get_terrain_factor(pos: np.ndarray) -> float:
    """
    Calculate the terrain slowdown factor based on the given position.
    
    Parameters:
        pos (np.ndarray): A 2D vector representing the position.
    
    Returns:
        float: The smallest modifier from the terrain zones that contain the position.
               Returns 1.0 if the position is not in any terrain zone.
    """
    factor = 1.0
    for zone in TERRAIN_ZONES:
        if zone.rect.collidepoint(pos[0], pos[1]):
            factor = min(factor, zone.modifier)
    return max(factor, 0.75)  # Ensure minimum speed is 75%

def distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.
    
    Parameters:
        a (np.ndarray): First point as a 2D vector.
        b (np.ndarray): Second point as a 2D vector.
        
    Returns:
        float: The Euclidean distance between points a and b.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])

# -------------------------------
# Resource Classes
# -------------------------------
@dataclass
class Plant:
    """
    A plant resource that creatures can forage.
    
    Attributes:
        pos (np.ndarray): The (x, y) position of the plant.
        active (bool): Whether the plant is currently available.
        regrow_timer (int): Frames remaining until the plant regrows.
    """
    pos: np.ndarray
    active: bool = True
    regrow_timer: int = 0

    def __init__(self, x: int, y: int):
        self.pos = np.array([x, y], dtype=float)
        self.active = True
        self.regrow_timer = 0

    def update(self) -> None:
        """
        Update the plant's state. If the plant is inactive, decrement the
        regrowth timer and reactivate the plant when the timer expires.
        """
        if not self.active:
            self.regrow_timer -= 1
            if self.regrow_timer <= 0:
                self.active = True

    def consume(self) -> None:
        """
        Mark the plant as consumed and set its regrowth timer.
        """
        self.active = False
        self.regrow_timer = PLANT_REGROW_TIME

@dataclass
class WaterSource:
    """
    A water resource that creatures can use.
    
    Attributes:
        pos (np.ndarray): The (x, y) position of the water source.
        active (bool): Whether the water source is currently available.
        regrow_timer (int): Frames remaining until the water source becomes available.
    """
    pos: np.ndarray
    active: bool = True
    regrow_timer: int = 0

    def __init__(self, x: int, y: int):
        self.pos = np.array([x, y], dtype=float)
        self.active = True
        self.regrow_timer = 0

    def update(self) -> None:
        """
        Update the water source's state. If the water source is inactive,
        decrement the regrowth timer and reactivate it when the timer expires.
        """
        if not self.active:
            self.regrow_timer -= 1
            if self.regrow_timer <= 0:
                self.active = True

    def consume(self) -> None:
        """
        Mark the water source as consumed and set its regrowth timer.
        """
        self.active = False
        self.regrow_timer = WATER_REGROW_TIME

# -------------------------------
# Base Creature Class
# -------------------------------
class Creature:
    def __init__(self, x, y, health, base_speed, hunger_duration):
        # Position and velocity (as float arrays)
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0], dtype=float)
        # Health and movement properties
        self.health = health
        self.max_health = health
        self.base_speed = base_speed
        # Hunger timer (in frames) and maximum hunger value for resets
        self.hunger_timer = hunger_duration
        self.max_hunger = hunger_duration
        self.energy = 0
        # Acceleration vector for force-based movement
        self.acceleration = np.array([0.0, 0.0], dtype=float)
        # Friction factor to gradually slow down movement over time
        self.friction = 0.99  # Less friction (was 0.98) for faster movement

    def update_position(self):
        """
        Updates the creature's position using its velocity and acceleration.
        Applies friction and bounces off the screen edges.
        """
        # Update velocity with acceleration and apply friction
        self.vel += self.acceleration * 2.0  # Doubled acceleration impact
        self.vel *= self.friction
        # Apply terrain effects with minimum speed
        terrain_factor = get_terrain_factor(self.pos)
        self.vel *= terrain_factor
        # Update position with doubled velocity
        self.pos += self.vel * 2.0
        # Reset acceleration for the next frame
        self.acceleration = np.array([0.0, 0.0], dtype=float)
        # Clamp position within screen boundaries
        self.pos[0] = np.clip(self.pos[0], 0, WIDTH)
        self.pos[1] = np.clip(self.pos[1], 0, HEIGHT)
        # Reverse velocity if hitting boundaries (simulate bounce)
        if self.pos[0] in (0, WIDTH):
            self.vel[0] *= -0.8
        if self.pos[1] in (0, HEIGHT):
            self.vel[1] *= -0.8

    def update_hunger(self):
        """
        Decreases the hunger timer. If hunger runs out, the creature takes damage.
        Returns True if the creature is starving.
        """
        self.hunger_timer -= 1
        if self.hunger_timer <= 0:
            self.take_damage(1)  # Creature loses 1 health per frame of starvation
            return True
        return False

    def update(self):
        """
        Updates the creature's state by processing movement, hunger, and 
        slowly recovering health if not starving.
        """
        self.update_position()
        starving = self.update_hunger()
        if not starving and self.health < self.max_health:
            # Slow recovery of health when not starving
            self.health = min(self.health + 0.05, self.max_health)

    def feed(self, energy_gain=10):
        """
        Resets the hunger timer and increases the creature's energy.
        
        Parameters:
            energy_gain (int): The amount of energy to add when feeding.
        """
        self.hunger_timer = self.max_hunger
        self.energy += energy_gain

    def take_damage(self, amount):
        """
        Reduces the creature's health by a given amount.
        
        Parameters:
            amount (float): The damage to inflict.
        """
        self.health -= amount
        if self.health < 0:
            self.health = 0

    def is_dead(self):
        """
        Returns True if the creature's health is 0 or less.
        """
        return self.health <= 0

    def apply_force(self, force):
        """
        Adds a force vector to the creature's acceleration.
        This can simulate influences like wind, repulsion, or other forces.
        
        Parameters:
            force (np.ndarray): A 2D force vector.
        """
        self.acceleration += force

    def get_status(self):
        """
        Returns a dictionary containing the creature's current status.
        Useful for debugging or logging.
        """
        return {
            "position": self.pos.tolist(),
            "velocity": self.vel.tolist(),
            "health": self.health,
            "energy": self.energy,
            "hunger_timer": self.hunger_timer
        }

# -------------------------------
# Goblin Class (Q–Learning, Grouping, Overrides to Attack Predators)
# All goblins are now red.
# -------------------------------

# (Assume WIDTH, HEIGHT, PLANT_REGROW_TIME, get_terrain_factor, and distance are defined elsewhere.)

class Goblin(Creature):
    def __init__(self, x: float, y: float,
                 q_weights1: np.ndarray = None, q_bias1: np.ndarray = None,
                 q_weights2: np.ndarray = None, q_bias2: np.ndarray = None,
                 alpha: float = 0.01, gamma: float = 0.9, epsilon: float = 0.1):
        """
        Initialize a Goblin creature with Q-learning parameters.
        
        Parameters:
            x, y (float): Initial position.
            q_weights1, q_bias1, q_weights2, q_bias2 (np.ndarray): Q-network parameters.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
        """
        super().__init__(x, y, 50, base_speed=6.0, hunger_duration=400)  # Doubled base_speed (was 3.0)
        self.energy = 0
        self.attack_damage = 15
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Q-network parameters; input dimension is now 3 (boar, plant, predator features)
        self.q_weights1 = q_weights1 if q_weights1 is not None else np.random.randn(3, 4)
        self.q_bias1    = q_bias1 if q_bias1 is not None else np.random.randn(4)
        self.q_weights2 = q_weights2 if q_weights2 is not None else np.random.randn(4, 2)
        self.q_bias2    = q_bias2 if q_bias2 is not None else np.random.randn(2)
        self.exp_buffer = []  # Experience replay buffer
        self.buffer_size = 20
        self.batch_size = 4
        self.last_state = np.zeros(3)  # Current state (3 features)
        self.last_action = 0
        self.target = None  # Current target (boar or plant)
        self.color = (255, 0, 0)

        # Action cooldowns (in frames) to prevent spamming actions
        self.reproduction_cooldown = 100
        self.attack_cooldown = 10
        self.forage_cooldown = 10

    def get_state(self, boars: list, plants: list, predators: list) -> np.ndarray:
        """
        Returns a 3-dimensional state vector:
         - boar_near: 1 if any boar is within 150 pixels, else 0.
         - plant_near: 1 if any active plant is within 150 pixels, else 0.
         - predator_near: 1 if any predator is within 150 pixels, else 0.
        """
        boar_near = 1.0 if any(distance(self.pos, b.pos) < 150 for b in boars) else 0.0
        plant_near = 1.0 if any(p.active and distance(self.pos, p.pos) < 150 for p in plants) else 0.0
        predator_near = 1.0 if any(distance(self.pos, p.pos) < 150 for p in predators) else 0.0
        return np.array([boar_near, plant_near, predator_near])

    def q_forward(self, state: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the Q-network.
        """
        hidden = np.tanh(np.dot(state, self.q_weights1) + self.q_bias1)
        return np.dot(hidden, self.q_weights2) + self.q_bias2

    def choose_action(self, boars: list, plants: list, predators: list, goblins: list) -> int:
        """
        Choose an action based on the Q-network output and nearby goblin targets.
        0: Hunt boars, 1: Forage plants.
        """
        state = self.get_state(boars, plants, predators)
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            q_vals = self.q_forward(state)
            action = int(np.argmax(q_vals))

        # Adjust action to reduce competition if other goblins already target a resource
        if action == 0:  # Hunting boars
            for g in goblins:
                if g is not self and g.target is not None and any(np.array_equal(g.target.pos, b.pos) for b in boars):
                    if random.random() < 0.5:
                        action = 1
                        break
        elif action == 1:  # Foraging plants
            for g in goblins:
                if g is not self and g.target is not None and any(np.array_equal(g.target.pos, p.pos) for p in plants):
                    if random.random() < 0.5:
                        action = 0
                        break

        self.last_state = state
        self.last_action = action
        return action

    def store_experience(self, reward: float, boars: list, plants: list, predators: list) -> None:
        """
        Store an experience tuple for later Q-network training.
        """
        next_state = self.get_state(boars, plants, predators)
        self.exp_buffer.append((self.last_state, self.last_action, reward, next_state))
        if len(self.exp_buffer) > self.buffer_size:
            self.exp_buffer.pop(0)

    def train_q_network(self) -> None:
        """
        Train the Q-network using a mini-batch from the experience buffer.
        Includes gradient clipping to stabilize training.
        """
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
            # Gradients for output layer
            grad_w2 += np.outer(hidden, np.eye(2)[action] * dloss_dq)
            grad_b2 += np.eye(2)[action] * dloss_dq
            # Gradients for hidden layer
            dtanh = 1 - hidden ** 2
            dhidden = (dloss_dq * self.q_weights2[:, action]) * dtanh
            grad_w1 += np.outer(state, dhidden)
            grad_b1 += dhidden

        # Apply gradient clipping
        grad_w1 = np.clip(grad_w1, -10, 10)
        grad_b1 = np.clip(grad_b1, -10, 10)
        grad_w2 = np.clip(grad_w2, -10, 10)
        grad_b2 = np.clip(grad_b2, -10, 10)

        # Update weights
        self.q_weights1 -= self.alpha * (grad_w1 / self.batch_size)
        self.q_bias1    -= self.alpha * (grad_b1 / self.batch_size)
        self.q_weights2 -= self.alpha * (grad_w2 / self.batch_size)
        self.q_bias2    -= self.alpha * (grad_b2 / self.batch_size)

        # Save updated parameters to q_params.json
        updated_params = {
            "q_weights1": self.q_weights1.tolist(),
            "q_bias1": self.q_bias1.tolist(),
            "q_weights2": self.q_weights2.tolist(),
            "q_bias2": self.q_bias2.tolist()
        }
        with open("q_params.json", "w") as f:
            json.dump(updated_params, f)

    def save_q_params(self, filename: str) -> None:
        """
        Save the Q-network parameters to a JSON file.
        """
        params = {
            'q_weights1': self.q_weights1.tolist(),
            'q_bias1': self.q_bias1.tolist(),
            'q_weights2': self.q_weights2.tolist(),
            'q_bias2': self.q_bias2.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(params, f)

    def load_q_params(self, filename: str) -> None:
        """
        Load Q-network parameters from a JSON file.
        """
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                params = json.load(f)
                self.q_weights1 = np.array(params['q_weights1'])
                self.q_bias1 = np.array(params['q_bias1'])
                self.q_weights2 = np.array(params['q_weights2'])
                self.q_bias2 = np.array(params['q_bias2'])

    def compute_separation_force(self, goblins: list) -> np.ndarray:
        """
        Compute a separation force to avoid clustering with other goblins.
        """
        separation = np.array([0.0, 0.0])
        count = 0
        for other in goblins:
            if other is not self:
                d = distance(self.pos, other.pos)
                if d < 20:
                    separation += (self.pos - other.pos) / (d + 1e-5)
                    count += 1
        return (separation / count) if count > 0 else separation

    def compute_predator_repulsion(self, predators: list) -> np.ndarray:
        """
        Compute an additional repulsion force from nearby predators.
        """
        repulsion = np.array([0.0, 0.0])
        count = 0
        for p in predators:
            d = distance(self.pos, p.pos)
            if d < 30:
                repulsion += (self.pos - p.pos) / (d + 1e-5)
                count += 1
        return (repulsion / count) if count > 0 else repulsion

    def update(self, boars: list, goblins: list, plants: list, predators: list, speed_global: float = 1.0) -> None:
        """
        Update the goblin's state:
         - If a predator is very close, decide whether to attack or flee.
         - Otherwise, choose an action via Q-learning and move toward the target.
         - Apply separation and predator repulsion forces.
         - Update position, train the Q-network, and decay epsilon.
        """
        # Decrement action cooldowns if active
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        if self.forage_cooldown > 0:
            self.forage_cooldown -= 1

        # Enhanced predator handling: decide whether to attack or flee based on energy.
        predator_target = None
        min_pd = float('inf')
        for p in predators:
            d = distance(self.pos, p.pos)
            if d < min_pd:
                min_pd = d
                predator_target = p

        if predator_target is not None and min_pd < 50:
            # If close to a predator, decide to attack if energy is high, else flee.
            if self.energy >= 30:
                direction = predator_target.pos - self.pos  # Attack
            else:
                direction = self.pos - predator_target.pos  # Flee
            norm = np.linalg.norm(direction) + 1e-5
            self.vel = (direction / norm) * self.base_speed * speed_global
            self.target = predator_target  # Temporarily set as target
        else:
            # Use Q-learning to choose action.
            action = self.choose_action(boars, plants, predators, goblins)  # 0: hunt boars, 1: forage plants
            if action == 0:
                # Target the nearest boar.
                closest = None
                min_d = float('inf')
                for b in boars:
                    d = distance(self.pos, b.pos)
                    if d < min_d:
                        min_d = d
                        closest = b
                if closest:
                    direction = closest.pos - self.pos
                    norm = np.linalg.norm(direction) + 1e-5
                    move = direction / norm
                    self.target = closest
                else:
                    move = np.array([0.0, 0.0])
                    self.target = None
                self.vel = move * self.base_speed * speed_global
            else:
                # Target the nearest active plant.
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
                    move = direction / norm
                    self.target = closest
                else:
                    move = np.array([0.0, 0.0])
                    self.target = None
                self.vel = move * self.base_speed * speed_global

        # Apply separation and extra predator repulsion forces.
        self.vel += self.compute_separation_force(goblins) * 0.5
        self.vel += self.compute_predator_repulsion(predators) * 1.0

        # Adjust velocity based on terrain effects.
        self.vel *= get_terrain_factor(self.pos)
        self.update_position()
        self.train_q_network()

        # Decay exploration rate gradually.
        self.epsilon = max(0.01, self.epsilon * 0.99999)

    def try_attack(self, boars: list, predators: list, plants: list) -> None:
        """
        Attempt to attack boars and predators if within range.
        Uses a cooldown to prevent spamming the attack.
        """
        if self.attack_cooldown > 0:
            return

        for b in boars[:]:
            if distance(self.pos, b.pos) < 10:
                b.health -= self.attack_damage
                if b.health <= 0:
                    self.energy += 50
                    self.store_experience(50, boars, plants, predators)
                    boars.remove(b)
                self.attack_cooldown = 10  # Set cooldown (in frames)
                break

        for p in predators[:]:
            if distance(self.pos, p.pos) < 10:
                p.health -= self.attack_damage
                if p.health <= 0:
                    self.energy += 50
                    self.store_experience(50, boars, plants, predators)
                    predators.remove(p)
                self.attack_cooldown = 10
                break

    def try_forage(self, plants: list, boars: list, predators: list) -> None:
        """
        Attempt to forage a plant if within range.
        Uses a cooldown to prevent repeated foraging actions.
        """
        if self.forage_cooldown > 0:
            return

        for p in plants:
            if p.active and distance(self.pos, p.pos) < 10:
                self.energy += 20
                self.hunger_timer = 400  # Reset hunger timer
                self.store_experience(20, boars, plants, predators)
                p.active = False
                p.regrow_timer = PLANT_REGROW_TIME
                self.forage_cooldown = 10
                break

    def try_reproduce(self, goblin_list: list) -> None:
        """
        Attempt to reproduce with a nearby goblin if both have enough energy.
        A reproduction cooldown prevents immediate successive reproductions.
        """
        reproduction_threshold = 60  # Energy threshold for reproduction
        if self.energy < reproduction_threshold or self.reproduction_cooldown > 0:
            return

        for other in goblin_list:
            if other is not self and other.energy >= reproduction_threshold and distance(self.pos, other.pos) < 20:
                new_x = (self.pos[0] + other.pos[0]) / 2 + random.uniform(-5, 5)
                new_y = (self.pos[1] + other.pos[1]) / 2 + random.uniform(-5, 5)
                new_base_speed = max(2.0, (self.base_speed + other.base_speed) / 2 + random.gauss(0, 0.1))
                new_attack_damage = max(5, (self.attack_damage + other.attack_damage) / 2 + random.gauss(0, 0.5))
                new_q_w1 = (self.q_weights1 + other.q_weights1) / 2 + np.random.randn(*self.q_weights1.shape) * 0.01
                new_q_b1 = (self.q_bias1 + other.q_bias1) / 2 + np.random.randn(*self.q_bias1.shape) * 0.01
                new_q_w2 = (self.q_weights2 + other.q_weights2) / 2 + np.random.randn(*self.q_weights2.shape) * 0.01
                new_q_b2 = (self.q_bias2 + other.q_bias2) / 2 + np.random.randn(*self.q_bias2.shape) * 0.01
                new_epsilon = max(0.01, (self.epsilon + other.epsilon) / 2 + random.gauss(0, 0.005))
                child = Goblin(new_x, new_y,
                               q_weights1=new_q_w1, q_bias1=new_q_b1,
                               q_weights2=new_q_w2, q_bias2=new_q_b2,
                               alpha=self.alpha, gamma=self.gamma, epsilon=new_epsilon)
                child.base_speed = new_base_speed
                child.attack_damage = new_attack_damage
                self.energy -= reproduction_threshold / 2
                other.energy -= reproduction_threshold / 2
                goblin_list.append(child)
                # Set reproduction cooldowns for both parents.
                self.reproduction_cooldown = 100
                other.reproduction_cooldown = 100
                break

# -------------------------------
# Boar Class (Beast)
# -------------------------------
class Boar(Creature):
    def __init__(self, x: float, y: float, species: str = None, action_weights: dict = None):
        """
        Initialize a Boar creature.

        Parameters:
            x (float): Initial x-position.
            y (float): Initial y-position.
            species (str, optional): Species identifier; defaults to a random choice between "boarA" and "boarB".
            action_weights (dict, optional): Weights for behavior decisions. Keys should include
                                             "forage", "wander", and "flee". Defaults to equal weights.
        """
        super().__init__(x, y, 100, base_speed=4.0, hunger_duration=900)  # Doubled base_speed (was 1.5)
        self.vel = np.array([0.0, 0.0], dtype=np.float64)
        self.foraging_counter: int = 10
        self.reproduction_weight: float = 1.0
        self.state: str = "wandering"
        self.species: str = species if species else random.choice(["boarA", "boarB"])
        self.action_weights: dict = action_weights if action_weights else {"forage": 1.0, "wander": 1.0, "flee": 1.0}
        self.attack_damage: int = 3
        self.reproduction_cooldown: int = 100  # Frames before the boar can reproduce again

    def update(self, goblins: list, plants: list, speed_global: float = 1.0) -> None:
        """
        Update the boar's behavior based on its environment.

        - If an active plant is nearby, decide to forage or wander based on action weights.
        - If goblin threats are near, flee by moving away from the average position of nearby goblins.
        - Otherwise, wander randomly.
        - Applies terrain slowdown and updates the boar's position.

        Parameters:
            goblins (list): List of goblin instances (potential threats).
            plants (list): List of plant instances (food sources).
            speed_global (float): A global speed modifier.
        """
        # Decrement reproduction cooldown if active.
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1

        plant_close = None
        for p in plants:
            if p.active and distance(self.pos, p.pos) < 15:
                plant_close = p
                break

        if plant_close:
            # Decide between foraging and wandering based on weighted probability.
            total_weight = self.action_weights["forage"] + self.action_weights["wander"]
            forage_probability = self.action_weights["forage"] / total_weight
            if random.random() < forage_probability:
                self.state = "foraging"
                self.foraging_counter += 1
                self.vel = np.array([0.0, 0.0])
            else:
                self.state = "wandering"
                rand_dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                norm = np.linalg.norm(rand_dir) + 1e-5
                self.vel = (rand_dir / norm) * self.base_speed * speed_global
                self.foraging_counter = max(0, self.foraging_counter - 1)
        else:
            # Check for nearby goblins as threats.
            threats = [g.pos for g in goblins if distance(self.pos, g.pos) < 100]
            if threats:
                self.state = "fleeing"
                avg_threat = np.mean(threats, axis=0)
                direction = self.pos - avg_threat
                norm = np.linalg.norm(direction) + 1e-5
                self.vel = (direction / norm) * 3 * speed_global
                self.foraging_counter = 0
            else:
                self.state = "wandering"
                rand_dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                norm = np.linalg.norm(rand_dir) + 1e-5
                self.vel = (rand_dir / norm) * self.base_speed * speed_global
                self.foraging_counter = max(0, self.foraging_counter - 1)

        # Ensure velocity is float64 and adjust by terrain factor.
        self.vel = self.vel.astype(np.float64)
        self.vel *= float(get_terrain_factor(self.pos))
        self.update_position()

    def try_attack(self, goblins: list) -> None:
        """
        Attempt to attack a goblin if within range.

        If a goblin is attacked and its health drops to 0 or below, the boar's reproduction weight increases.
        """
        for g in goblins[:]:
            if distance(self.pos, g.pos) < 10:
                g.health -= self.attack_damage
                if g.health <= 0:
                    self.reproduction_weight += 0.5
                break

    def try_reproduce(self, boar_list: list) -> None:
        """
        Attempt to reproduce with a nearby boar of the same species.

        Reproduction occurs if either the foraging counter reaches a threshold (BOAR_FORAGE_DURATION)
        or the reproduction weight is high enough. A reproduction cooldown prevents immediate repeat reproduction.

        The offspring inherits a combination of the parent's action weights with a small mutation.
        """
        if (self.foraging_counter >= BOAR_FORAGE_DURATION or self.reproduction_weight >= 2.0) and self.reproduction_cooldown == 0:
            for other in boar_list:
                if (other is not self and other.species == self.species and other.reproduction_cooldown == 0 and
                    (other.foraging_counter >= BOAR_FORAGE_DURATION or other.reproduction_weight >= 2.0) and 
                    distance(self.pos, other.pos) < 20):
                    
                    new_x = (self.pos[0] + other.pos[0]) / 2 + random.uniform(-5, 5)
                    new_y = (self.pos[1] + other.pos[1]) / 2 + random.uniform(-5, 5)
                    # Combine the parent's action weights with a slight mutation.
                    new_weights = {k: (self.action_weights[k] + other.action_weights[k]) / 2 + random.gauss(0, 0.01)
                                   for k in self.action_weights}
                    child = Boar(new_x, new_y, species=self.species, action_weights=new_weights)
                    # Reset foraging counters and reproduction weights for both parents.
                    self.foraging_counter = 0
                    other.foraging_counter = 0
                    self.reproduction_weight = 1.0
                    other.reproduction_weight = 1.0
                    # Set reproduction cooldown for both parents.
                    self.reproduction_cooldown = 100
                    other.reproduction_cooldown = 100
                    boar_list.append(child)
                    break

    def try_forage(self, plants: list) -> None:
        """
        Attempt to forage a plant if one is within range.

        On successful foraging:
            - The hunger timer is reset.
            - The boar gains energy.
            - The plant is marked as consumed and its regrowth timer is started.
        """
        for p in plants:
            if p.active and distance(self.pos, p.pos) < 10:
                self.hunger_timer = 900  # Reset hunger timer.
                self.energy += 10        # Gain energy from foraging.
                p.active = False
                p.regrow_timer = PLANT_REGROW_TIME
                break

# -------------------------------
# Scavenger Class (Giant Spiders - Very Solitary)
# -------------------------------
class Scavenger(Creature):
    def __init__(self, x: float, y: float, species: str = None, action_weights: dict = None):
        """
        Initialize a Scavenger creature.

        Parameters:
            x (float): Initial x-coordinate.
            y (float): Initial y-coordinate.
            species (str, optional): Species identifier; defaults to a random choice between "spiderA" and "spiderB".
            action_weights (dict, optional): Weights for behavior decisions (keys: "steal" and "forage").
        """
        super().__init__(x, y, 40, base_speed=5.0, hunger_duration=750)  # Doubled base_speed (was 2.5)
        self.energy: float = 0
        self.steal_amount: int = 20
        self.species: str = species if species is not None else random.choice(["spiderA", "spiderB"])
        self.action_weights: dict = action_weights if action_weights is not None else {"steal": 1.0, "forage": 1.0}
        self.attack_damage: int = 3
        # Two-layer network parameters for processing movement when stealing.
        self.weights1: np.ndarray = np.random.randn(2, 4)
        self.bias1: np.ndarray = np.random.randn(4)
        self.weights2: np.ndarray = np.random.randn(4, 2)
        self.bias2: np.ndarray = np.random.randn(2)
        self.reproduction_cooldown: int = 100  # Cooldown (in frames) to prevent immediate successive reproduction

    def decide_action(self, goblins: list, plants: list) -> str:
        """
        Decide the scavenger's action based on nearby goblins and available plants.

        Returns:
            str: "steal" if conditions favor stealing from a goblin, otherwise "forage".
        """
        targets = [g for g in goblins if g.energy > 20 and distance(self.pos, g.pos) < 150]
        if targets:
            total = self.action_weights["steal"] + self.action_weights["forage"]
            if random.random() < (self.action_weights["steal"] / total):
                return "steal"
        return "forage"

    def decide_movement_steal(self, goblins: list) -> np.ndarray:
        """
        Determine the direction to move when attempting to steal.

        Returns:
            np.ndarray: A unit vector in the direction of the closest goblin with sufficient energy.
        """
        target = None
        min_d = float('inf')
        for g in goblins:
            if g.energy > 20:
                d = distance(self.pos, g.pos)
                if d < min_d:
                    min_d = d
                    target = g
        if target is not None:
            direction = target.pos - self.pos
            norm = np.linalg.norm(direction) + 1e-5
            return direction / norm
        return np.array([0, 0])

    def decide_movement_forage(self, plants: list) -> np.ndarray:
        """
        Determine the direction to move when foraging for plants.

        Returns:
            np.ndarray: A unit vector pointing toward the nearest active plant.
        """
        closest = None
        min_d = float('inf')
        for p in plants:
            if p.active:
                d = distance(self.pos, p.pos)
                if d < min_d:
                    min_d = d
                    closest = p
        if closest is not None:
            direction = closest.pos - self.pos
            norm = np.linalg.norm(direction) + 1e-5
            return direction / norm
        return np.array([0, 0])

    def update(self, goblins: list, plants: list, scavengers: list, speed_global: float = 1.0) -> None:
        """
        Update the scavenger's movement and state.

        - Decides between stealing and foraging.
        - For "steal": uses a two-layer network to process the movement direction.
        - For "forage": moves toward the nearest active plant.
        - Applies a separation force from nearby scavengers.
        - Adjusts velocity based on terrain and updates position.

        Parameters:
            goblins (list): List of goblin instances.
            plants (list): List of plant instances.
            scavengers (list): List of other scavenger instances.
            speed_global (float): Global speed modifier.
        """
        # Decrement reproduction cooldown if active.
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1

        action = self.decide_action(goblins, plants)
        if action == "steal":
            move_direction = self.decide_movement_steal(goblins)
            hidden = np.tanh(np.dot(move_direction, self.weights1) + self.bias1)
            output = np.tanh(np.dot(hidden, self.weights2) + self.bias2)
            self.vel = output * self.base_speed * speed_global
        else:
            move_direction = self.decide_movement_forage(plants)
            self.vel = move_direction * self.base_speed * speed_global

        # Separation: avoid clustering with other scavengers.
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
            self.vel += separation * 2.0  # Apply strong repulsion force.

        self.vel *= float(get_terrain_factor(self.pos))
        self.update_position()

    def try_steal(self, goblins: list) -> None:
        """
        Attempt to steal energy from a nearby goblin.

        If a goblin with positive energy is within 10 pixels, steal an amount up to self.steal_amount,
        increase self.energy, and boost the "steal" action weight slightly.
        """
        for g in goblins:
            if g.energy > 0 and distance(self.pos, g.pos) < 10:
                stolen = min(self.steal_amount, g.energy)
                g.energy -= stolen
                self.energy += stolen
                self.action_weights["steal"] *= 1.1
                break

    def try_forage(self, plants: list) -> None:
        """
        Attempt to forage an active plant if within range.

        On successful foraging:
            - Reset the scavenger's hunger timer.
            - Mark the plant as consumed and start its regrowth timer.
        """
        for p in plants:
            if p.active and distance(self.pos, p.pos) < 10:
                self.hunger_timer = 750  # Reset hunger timer.
                p.active = False
                p.regrow_timer = PLANT_REGROW_TIME
                break

    def try_reproduce(self, scavenger_list: list) -> None:
        """
        Attempt to reproduce with another scavenger of the same species if both have enough energy.

        Reproduction reduces both parents' energy and produces a new scavenger with a combination
        (and slight mutation) of the parents' action weights. A reproduction cooldown prevents immediate successive reproduction.
        """
        threshold = 120  # Energy threshold for reproduction.
        if self.energy < threshold or self.reproduction_cooldown > 0:
            return

        for other in scavenger_list:
            if (other is not self and other.species == self.species and 
                other.energy >= threshold and other.reproduction_cooldown == 0 and 
                distance(self.pos, other.pos) < 20):
                new_x = (self.pos[0] + other.pos[0]) / 2 + random.uniform(-5, 5)
                new_y = (self.pos[1] + other.pos[1]) / 2 + random.uniform(-5, 5)
                new_aw = {k: (self.action_weights[k] + other.action_weights[k]) / 2 + random.gauss(0, 0.01)
                          for k in self.action_weights}
                child = Scavenger(new_x, new_y, species=self.species, action_weights=new_aw)
                self.energy -= threshold / 2
                other.energy -= threshold / 2
                scavenger_list.append(child)
                # Set reproduction cooldown for both parents.
                self.reproduction_cooldown = 100
                other.reproduction_cooldown = 100
                break

# -------------------------------
# Predator Class (Wolves) – Now also target scavengers
# Also predators maintain solitary spacing.
# -------------------------------
class Predator(Creature):
    def __init__(self, x: float, y: float, species: str = None):
        """
        Initialize a Predator instance.
        
        Parameters:
            x (float): Initial x-coordinate.
            y (float): Initial y-coordinate.
            species (str, optional): Predator species identifier; if not provided, one is chosen at random.
        """
        super().__init__(x, y, 80, base_speed=7.0, hunger_duration=800)  # Doubled base_speed (was 3.0)
        self.energy: float = 0
        self.species: str = species if species is not None else random.choice(["wolfA", "wolfB"])
        self.attack_damage: int = 20
        self.attack_cooldown: int = 10        # Cooldown (in frames) to prevent repeated attacks
        self.reproduction_cooldown: int = 100  # Cooldown (in frames) to prevent immediate successive reproduction

    def update(self, goblins: list, boars: list, scavengers: list, predators: list, speed_global: float = 1.0) -> None:
        """
        Update the predator's state: chase the nearest prey (from goblins, boars, or scavengers) if available;
        otherwise, wander randomly. Also applies a separation force from nearby predators.
        
        Parameters:
            goblins (list): List of goblin instances (prey).
            boars (list): List of boar instances (prey).
            scavengers (list): List of scavenger instances (prey).
            predators (list): List of other predator instances.
            speed_global (float): A global speed multiplier.
        """
        # Decrement cooldown timers if active.
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1

        # Chase the nearest prey.
        prey_list = goblins + boars + scavengers
        target = None
        min_d = float('inf')
        goblin_weight = 1.5  # Increase this value to make goblins less attractive

        for p in prey_list:
            d = distance(self.pos, p.pos)
            if isinstance(p, Goblin):
                d *= goblin_weight  # Apply weighting factor to goblins
            if d < min_d:
                min_d = d
                target = p

        if target:
            direction = target.pos - self.pos
            norm = np.linalg.norm(direction) + 1e-5
            self.vel = (direction / norm) * self.base_speed * speed_global
        else:
            # No prey found: wander randomly.
            rand_dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
            norm = np.linalg.norm(rand_dir) + 1e-5
            self.vel = (rand_dir / norm) * self.base_speed * speed_global

        # Separation: predators stay solitary.
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

        # Adjust velocity based on terrain effects.
        self.vel *= get_terrain_factor(self.pos)
        self.update_position()

    def try_attack(self, goblins: list, boars: list, scavengers: list) -> None:
        """
        Attempt to attack nearby prey (goblins, boars, or scavengers) if within range.
        Uses an attack cooldown to prevent rapid consecutive attacks.
        
        Parameters:
            goblins (list): List of goblin instances.
            boars (list): List of boar instances.
            scavengers (list): List of scavenger instances.
        """
        if self.attack_cooldown > 0:
            return

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
                self.attack_cooldown = 10  # Set attack cooldown (in frames)
                break

    def try_reproduce(self, predator_list: list) -> None:
        """
        Attempt to reproduce with another predator of the same species if both have sufficient energy.
        A reproduction cooldown prevents immediate successive reproduction.
        
        Parameters:
            predator_list (list): List of predator instances.
        """
        threshold = 200
        if self.energy < threshold or self.reproduction_cooldown > 0:
            return

        for other in predator_list:
            if (other is not self and other.species == self.species and 
                other.energy >= threshold and other.reproduction_cooldown == 0):
                if distance(self.pos, other.pos) < 20:
                    new_x = (self.pos[0] + other.pos[0]) / 2 + random.uniform(-5, 5)
                    new_y = (self.pos[1] + other.pos[1]) / 2 + random.uniform(-5, 5)
                    child = Predator(new_x, new_y, species=self.species)
                    self.energy -= threshold / 2
                    other.energy -= threshold / 2
                    self.reproduction_cooldown = 100  # Set reproduction cooldown for both parents.
                    other.reproduction_cooldown = 100
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

    # Load sound effects (if available)
    try:
        attack_sound = pygame.mixer.Sound("attack.wav")
        reproduce_sound = pygame.mixer.Sound("reproduce.wav")
    except Exception as e:
        print("Sound loading error:", e)
        attack_sound = None
        reproduce_sound = None

    # Open simulation log file for appending data
    if not os.path.exists("simulation_log.txt"):
        with open("simulation_log.txt", "w") as f:
            f.write("timestamp,goblins,boars,scavengers,predators,plants\n")
    log_file = open("simulation_log.txt", "a")
    
    # Create default q_params file if it doesn't exist
    q_params_file = "q_params.json"
    if not os.path.exists(q_params_file):
        default_params = {
            "q_weights1": [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
            "q_bias1": [0.1, 0.1, 0.1, 0.1],
            "q_weights2": [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]],
            "q_bias2": [0.1, 0.1]
        }
        with open(q_params_file, "w") as f:
            json.dump(default_params, f)

    # Initialize populations
    goblins = [Goblin(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(18)]
    for g in goblins:
        g.load_q_params(q_params_file)
    boars = [Boar(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(27)]
    scavengers = [Scavenger(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(18)]
    predators = [Predator(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(4)]
    plants = [Plant(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(62)]

    frame_count = 0
    speed_factor = 8.0  # Doubled global speed factor (was 1.0)
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0  # Get delta time in seconds
        frame_count += 1

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Update Resources ---
        for plant in plants:
            plant.update()
        # Spawn additional plants if needed
        if len(plants) < 50 and random.random() < 0.02:
            plants.append(Plant(random.randint(0, WIDTH), random.randint(0, HEIGHT)))

        # --- Update Creatures ---
        # Update Goblins
        for g in goblins[:]:
            g.update(boars, goblins, plants, predators, speed_global=speed_factor * dt)
            g.try_attack(boars, predators, plants)
            g.try_forage(plants, boars, predators)
            g.try_reproduce(goblins)
            if g.is_dead() or g.update_hunger():
                goblins.remove(g)

        # Update Boars
        for b in boars[:]:
            b.update(goblins, plants, speed_global=speed_factor * dt)
            b.try_attack(goblins)
            b.try_reproduce(boars)
            b.try_forage(plants)
            if b.is_dead() or b.update_hunger():
                boars.remove(b)

        # Update Scavengers
        for s in scavengers[:]:
            s.update(goblins, plants, scavengers, speed_global=speed_factor * dt)
            s.try_steal(goblins)
            s.try_forage(plants)
            s.try_reproduce(scavengers)
            if s.is_dead() or s.update_hunger():
                scavengers.remove(s)

        # Update Predators
        for p in predators[:]:
            p.update(goblins, boars, scavengers, predators, speed_global=speed_factor * dt)
            p.try_attack(goblins, boars, scavengers)
            p.try_reproduce(predators)
            if p.is_dead() or p.update_hunger():
                predators.remove(p)

        # --- External Spawning ---
        if len(scavengers) < 3 and random.random() < (len(goblins) / 100.0):
            edge = random.randint(0, 7)
            if edge == 0:
                x, y = -10, random.randint(0, HEIGHT)
            elif edge == 1:
                x, y = WIDTH + 10, random.randint(0, HEIGHT)
            elif edge == 2:
                x, y = random.randint(0, WIDTH), -10
            else:
                x, y = random.randint(0, WIDTH), HEIGHT + 10
            scavengers.append(Scavenger(x, y))

        if len(boars) < 3 and random.random() < (len(plants) / 100.0):
            edge = random.randint(0, 10)
            if edge == 0:
                x, y = -10, random.randint(0, HEIGHT)
            elif edge == 1:
                x, y = WIDTH + 10, random.randint(0, HEIGHT)
            elif edge == 2:
                x, y = random.randint(0, WIDTH), -10
            else:
                x, y = random.randint(0, WIDTH), HEIGHT + 10
            boars.append(Boar(x, y))

        if len(predators) < 3 and random.random() < ((len(boars) + len(goblins)) / 200.0):
            edge = random.randint(0, 4)
            if edge == 0:
                x, y = -10, random.randint(0, HEIGHT)
            elif edge == 1:
                x, y = WIDTH + 10, random.randint(0, HEIGHT)
            elif edge == 2:
                x, y = random.randint(0, WIDTH), -10
            else:
                x, y = random.randint(0, WIDTH), HEIGHT + 10
            predators.append(Predator(x, y))

        # --- Drawing ---
        screen.fill((50, 50, 80))  # Background color
        # Draw terrain zones (using the TerrainZone dataclass)
        for zone in TERRAIN_ZONES:
            pygame.draw.rect(screen, (30, 60, 30), zone.rect, 2)
        # Draw plants
        for plant in plants:
            if plant.active:
                pygame.draw.circle(screen, (0, 255, 0),
                                   (int(plant.pos[0]), int(plant.pos[1])), 5)
        # Draw goblins with a velocity line
        for g in goblins:
            pygame.draw.circle(screen, g.color,
                               (int(g.pos[0]), int(g.pos[1])), 5)
            end = (int(g.pos[0] + g.vel[0]*5), int(g.pos[1] + g.vel[1]*5))
            pygame.draw.line(screen, (0, 0, 0),
                             (int(g.pos[0]), int(g.pos[1])), end, 2)
        # Draw boars (color changes if foraging)
        for b in boars:
            col = (139, 69, 19) if b.state != "foraging" else (0, 0, 255)
            pygame.draw.circle(screen, col,
                               (int(b.pos[0]), int(b.pos[1])), 8)
        # Draw scavengers
        for s in scavengers:
            pygame.draw.circle(screen, (128, 0, 128),
                               (int(s.pos[0]), int(s.pos[1])), 6)
        # Draw predators
        for p in predators:
            pygame.draw.circle(screen, (100, 100, 100),
                               (int(p.pos[0]), int(p.pos[1])), 7)

        # Display simulation metrics
        metrics = (f"Goblins: {len(goblins)}  Boars: {len(boars)}  "
                   f"Scavengers: {len(scavengers)}  Predators: {len(predators)}  "
                   f"Plants: {len(plants)}")
        text_surface = font.render(metrics, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        pygame.display.flip()

        # --- Logging ---
        if frame_count % 30 == 0:  # Log every second (at 30 FPS)
            log_line = f"{time.time()},{len(goblins)},{len(boars)},{len(scavengers)},{len(predators)},{len(plants)}\n"
            log_file.write(log_line)
            log_file.flush()

        # --- Restart Goblins if Extinct ---
        if len(goblins) == 0:
            # Save Q-learning parameters if available before restarting
            # (Could be improved by saving the best performing goblin's parameters)
            if goblins:
                goblins[0].save_q_params(q_params_file)
            goblins = [Goblin(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(10)]
            for g in goblins:
                g.load_q_params(q_params_file)

    log_file.close()
    pygame.quit()


if __name__ == "__main__":
    main()
