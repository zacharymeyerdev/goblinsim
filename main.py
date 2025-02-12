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

def resolve_collisions(obj, others, min_distance):
    """
    Adjust obj.pos so that it is not closer than min_distance to any object in 'others'.
    """
    push = np.array([0.0, 0.0])
    count = 0
    for other in others:
        if other is obj:
            continue
        d = distance(obj.pos, other.pos)
        if d < min_distance:
            # Compute a normalized vector pointing away from the other object.
            direction = (obj.pos - other.pos) / (d + 1e-5)
            push += (min_distance - d) * direction
            count += 1
    if count > 0:
        obj.pos += push / count

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
    def __init__(self, x: int, y: int):
        self.pos = np.array([x, y], dtype=float)
        self.active = True
        self.regrow_timer = 0

    def update(self, all_plants: list = None, all_creatures: list = None) -> None:
        """
        Update the plant's state.
        
        If the plant is inactive, decrement the regrowth timer and reactivate it when the timer expires.
        Additionally, if a list of all plants is provided, resolve collisions so that no two plants overlap.
        If a list of all creatures is provided, resolve collisions to ensure no creature occupies the same pixels.
        """
        if not self.active:
            self.regrow_timer -= 1
            if self.regrow_timer <= 0:
                self.active = True
        
        # Resolve collisions with other plants.
        if all_plants is not None:
            resolve_collisions(self, all_plants, 10)
        
        # Resolve collisions with creatures (goblins, boars, scavengers, predators).
        if all_creatures is not None:
            resolve_collisions(self, all_creatures, 10)

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
        Initialize a Goblin creature with Q-learning parameters and additional behaviors
        such as group aggression, flocking, ambush, territoriality, energy sharing,
        dynamic aggression, communication, resting state, and collision resolution.
        """
        super().__init__(x, y, 50, base_speed=6.0, hunger_duration=400)
        self.energy = 0
        self.attack_damage = 15
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Q-network parameters; input dimension is 3 (boar, plant, predator features)
        self.q_weights1 = q_weights1 if q_weights1 is not None else np.random.randn(3, 4)
        self.q_bias1    = q_bias1 if q_bias1 is not None else np.random.randn(4)
        self.q_weights2 = q_weights2 if q_weights2 is not None else np.random.randn(4, 2)
        self.q_bias2    = q_bias2 if q_bias2 is not None else np.random.randn(2)
        self.exp_buffer = []  # Experience replay buffer
        self.buffer_size = 20
        self.batch_size = 4
        self.last_state = np.zeros(3)  # Current state (3 features)
        self.last_action = 0
        self.target = None  # Current target (boar, plant, or predator)
        self.color = (255, 0, 0)

        # Action cooldowns (in frames)
        self.reproduction_cooldown = 100
        self.attack_cooldown = 10
        self.forage_cooldown = 10

        # Additional behavior attributes:
        self.fear = 1.0             # High fear at spawn (prevents immediate aggression)
        self.aggression = 0.1       # Dynamic aggression level (0.0 to 1.0)
        self.in_ambush_mode = False # Flag for ambush/stealth state
        self.threat_alert = False   # Flag set when a threat signal is received
        self.last_threat_position = None
        self.state = "active"       # Can be "active" or "resting"

    # --- Q-Learning Methods ---
    def get_state(self, boars: list, plants: list, predators: list) -> np.ndarray:
        boar_near = 1.0 if any(distance(self.pos, b.pos) < 150 for b in boars) else 0.0
        plant_near = 1.0 if any(p.active and distance(self.pos, p.pos) < 150 for p in plants) else 0.0
        predator_near = 1.0 if any(distance(self.pos, p.pos) < 150 for p in predators) else 0.0
        return np.array([boar_near, plant_near, predator_near])

    def q_forward(self, state: np.ndarray) -> np.ndarray:
        hidden = np.tanh(np.dot(state, self.q_weights1) + self.q_bias1)
        return np.dot(hidden, self.q_weights2) + self.q_bias2

    def choose_action(self, boars: list, plants: list, predators: list, goblins: list) -> int:
        """
        Choose an action:
          0 = Attack boars, 1 = Forage plants.
        If fear is high, force foraging regardless of group cues.
        Otherwise, use Q-learning and group conditions.
        """
        if self.fear > 0.5:
            return 1  # Forage when fear is high

        state = self.get_state(boars, plants, predators)
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            q_vals = self.q_forward(state)
            action = int(np.argmax(q_vals))
        
        # Check group conditions.
        group_count = sum(1 for g in goblins if g is not self and distance(self.pos, g.pos) < 50)
        boar_near = any(distance(self.pos, b.pos) < 150 for b in boars)
        plant_near = any(p.active and distance(self.pos, p.pos) < 150 for p in plants)
        
        # If group conditions favor aggression but plants are available,
        # force foraging 50% of the time.
        if group_count >= 1 and boar_near:
            if plant_near and random.random() < 0.5:
                action = 1
            else:
                action = 0
        self.last_state = state
        self.last_action = action
        return action

    def store_experience(self, reward: float, boars: list, plants: list, predators: list) -> None:
        next_state = self.get_state(boars, plants, predators)
        self.exp_buffer.append((self.last_state, self.last_action, reward, next_state))
        if len(self.exp_buffer) > self.buffer_size:
            self.exp_buffer.pop(0)

    def train_q_network(self) -> None:
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
            dtanh = 1 - hidden ** 2
            dhidden = (dloss_dq * self.q_weights2[:, action]) * dtanh
            grad_w1 += np.outer(state, dhidden)
            grad_b1 += dhidden

        grad_w1 = np.clip(grad_w1, -10, 10)
        grad_b1 = np.clip(grad_b1, -10, 10)
        grad_w2 = np.clip(grad_w2, -10, 10)
        grad_b2 = np.clip(grad_b2, -10, 10)

        self.q_weights1 -= self.alpha * (grad_w1 / self.batch_size)
        self.q_bias1    -= self.alpha * (grad_b1 / self.batch_size)
        self.q_weights2 -= self.alpha * (grad_w2 / self.batch_size)
        self.q_bias2    -= self.alpha * (grad_b2 / self.batch_size)

        updated_params = {
            "q_weights1": self.q_weights1.tolist(),
            "q_bias1": self.q_bias1.tolist(),
            "q_weights2": self.q_weights2.tolist(),
            "q_bias2": self.q_bias2.tolist()
        }
        with open("q_params.json", "w") as f:
            json.dump(updated_params, f)

    def save_q_params(self, filename: str) -> None:
        params = {
            'q_weights1': self.q_weights1.tolist(),
            'q_bias1': self.q_bias1.tolist(),
            'q_weights2': self.q_weights2.tolist(),
            'q_bias2': self.q_bias2.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(params, f)

    def load_q_params(self, filename: str) -> None:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                params = json.load(f)
                self.q_weights1 = np.array(params['q_weights1'])
                self.q_bias1 = np.array(params['q_bias1'])
                self.q_weights2 = np.array(params['q_weights2'])
                self.q_bias2 = np.array(params['q_bias2'])

    # --- Additional Behavior Methods ---
    def compute_separation_force(self, goblins: list) -> np.ndarray:
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
        repulsion = np.array([0.0, 0.0])
        count = 0
        for p in predators:
            d = distance(self.pos, p.pos)
            if d < 30:
                repulsion += (self.pos - p.pos) / (d + 1e-5)
                count += 1
        return (repulsion / count) if count > 0 else repulsion

    def compute_cohesion_force(self, goblins: list) -> np.ndarray:
        center = np.array([0.0, 0.0])
        count = 0
        for g in goblins:
            if g is not self and distance(self.pos, g.pos) < 50:
                center += g.pos
                count += 1
        if count > 0:
            center /= count
            return (center - self.pos) * 0.05
        return np.array([0.0, 0.0])

    def mark_territory(self, territory_map: dict):
        cell_size = 20
        pos_key = (int(self.pos[0] // cell_size), int(self.pos[1] // cell_size))
        territory_map[pos_key] = territory_map.get(pos_key, 0) + 1

    def check_territory(self, territory_map: dict) -> bool:
        cell_size = 20
        pos_key = (int(self.pos[0] // cell_size), int(self.pos[1] // cell_size))
        return territory_map.get(pos_key, 0) > 3

    def try_enter_ambush(self, nearby_terrain: list):
        if any(distance(self.pos, t) < 50 for t in nearby_terrain):
            self.in_ambush_mode = True
        else:
            self.in_ambush_mode = False

    def update_ambush_speed(self):
        if self.in_ambush_mode:
            self.base_speed = 2.0
        else:
            self.base_speed = 6.0

    def share_energy(self, goblins: list):
        if self.energy > 80:
            for g in goblins:
                if g is not self and distance(self.pos, g.pos) < 50 and g.energy < 40:
                    transfer = (self.energy - 80) * 0.5
                    g.energy += transfer
                    self.energy -= transfer

    def update_aggression(self, damage_taken: float = 0, successful_attack: bool = False):
        if successful_attack:
            self.aggression = min(1.0, self.aggression + 0.1)
        if damage_taken > 0:
            self.aggression = max(0.0, self.aggression - 0.05)

    def decide_target(self, boars: list, predators: list):
        if self.aggression > 0.7 and predators:
            return min(predators, key=lambda p: distance(self.pos, p.pos))
        elif boars:
            return min(boars, key=lambda b: distance(self.pos, b.pos))
        return None

    def signal_threat(self, goblins: list, threat_position: np.ndarray):
        for g in goblins:
            if g is not self and distance(self.pos, g.pos) < 100:
                g.receive_signal(threat_position)

    def receive_signal(self, threat_position: np.ndarray):
        self.threat_alert = True
        self.last_threat_position = threat_position

    def update_state(self):
        if self.health < 20 or self.energy < 20:
            self.state = "resting"
            self.base_speed = 2.0
            self.energy += 0.5  # Recovery rate
        else:
            self.state = "active"
            self.update_ambush_speed()

    def resolve_collisions(self, goblins: list) -> None:
        push = np.array([0.0, 0.0])
        count = 0
        min_distance = 10  # Minimum allowed distance between goblins
        for other in goblins:
            if other is not self:
                d = distance(self.pos, other.pos)
                if d < min_distance:
                    direction = (self.pos - other.pos) / (d + 1e-5)
                    push += (min_distance - d) * direction
                    count += 1
        if count > 0:
            self.pos += push / count

    # --- Main Update Method ---
    def update(self, boars: list, goblins: list, plants: list, predators: list,
               speed_global: float = 1.0, nearby_terrain: list = None, territory_map: dict = None) -> None:
        # Update internal state (resting) and ambush mode.
        self.update_state()
        if nearby_terrain is not None:
            self.try_enter_ambush(nearby_terrain)
        self.update_ambush_speed()

        # Share energy with nearby goblins.
        self.share_energy(goblins)

        # Mark territory if a map is provided.
        if territory_map is not None:
            self.mark_territory(territory_map)

        # Compute group forces.
        group_count = sum(1 for g in goblins if g is not self and distance(self.pos, g.pos) < 50)
        separation_force = self.compute_separation_force(goblins)
        cohesion_force = self.compute_cohesion_force(goblins)

        # Edge avoidance: push inward if near borders.
        edge_force = np.array([0.0, 0.0])
        margin = 20
        if self.pos[0] < margin:
            edge_force[0] += 1
        elif self.pos[0] > WIDTH - margin:
            edge_force[0] -= 1
        if self.pos[1] < margin:
            edge_force[1] += 1
        elif self.pos[1] > HEIGHT - margin:
            edge_force[1] -= 1
        edge_force *= 0.5

        # Threat communication.
        if self.threat_alert and self.last_threat_position is not None:
            threat_dir = self.last_threat_position - self.pos
            if np.linalg.norm(threat_dir) > 1e-5:
                threat_force = (threat_dir / (np.linalg.norm(threat_dir) + 1e-5)) * 0.5
            else:
                threat_force = np.array([0.0, 0.0])
        else:
            threat_force = np.array([0.0, 0.0])

        # Sum additional forces.
        self.vel += separation_force * 0.5 + cohesion_force * 0.5 + edge_force + threat_force

        # Check for immediate predator threat (within 50 pixels).
        predator_target = None
        min_pd = float('inf')
        for p in predators:
            d = distance(self.pos, p.pos)
            if d < min_pd:
                min_pd = d
                predator_target = p

        if predator_target is not None and min_pd < 50:
            if group_count >= 3 or self.energy >= 30 or self.aggression > 0.7:
                direction = predator_target.pos - self.pos  # Attack
            else:
                direction = self.pos - predator_target.pos  # Flee
            norm = np.linalg.norm(direction) + 1e-5
            self.vel = (direction / norm) * self.base_speed * speed_global
            self.target = predator_target
            self.signal_threat(goblins, predator_target.pos)
        else:
            # Choose action via Q-learning.
            action = self.choose_action(boars, plants, predators, goblins)
            if action == 0:
                # Attack boars: target nearest boar.
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
                # Forage: target nearest active plant.
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

        # Apply extra predator repulsion.
        self.vel += self.compute_predator_repulsion(predators) * 1.0

        # Adjust velocity based on terrain.
        self.vel *= get_terrain_factor(self.pos)
        self.update_position()

        # Resolve collisions so goblins do not overlap.
        self.resolve_collisions(goblins)

        self.train_q_network()
        self.epsilon = max(0.01, self.epsilon * 0.99999)

    # --- On-Contact Methods (Now integrated into update) ---
    def try_attack(self, boars: list, predators: list, plants: list) -> None:
        # Not used; on-contact damage is handled in update.
        pass

    def try_forage(self, plants: list, boars: list, predators: list) -> None:
        # Not used; on-contact foraging is handled in update.
        pass

    def try_reproduce(self, goblin_list: list) -> None:
        group_count = sum(1 for g in goblin_list if g is not self and distance(self.pos, g.pos) < 50)
        reproduction_threshold = 50 if group_count >= 1 else 60
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
        """
        super().__init__(x, y, 100, base_speed=4.0, hunger_duration=900)
        self.vel = np.array([0.0, 0.0], dtype=np.float64)
        self.foraging_counter = 10
        self.reproduction_weight = 1.0
        self.state = "wandering"
        self.species = species if species else random.choice(["boarA", "boarB"])
        self.action_weights = action_weights if action_weights else {"forage": 1.0, "wander": 1.0, "flee": 1.0}
        self.attack_damage = 3
        self.reproduction_cooldown = 100

    def update(self, goblins: list, plants: list, speed_global: float = 1.0):
        # Decrement reproduction cooldown.
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1

        # Decide behavior: if a plant is very close, forage; if goblins (threats) are nearby, flee; else wander.
        plant_close = None
        for p in plants:
            if p.active and distance(self.pos, p.pos) < 15:
                plant_close = p
                break

        if plant_close:
            self.state = "foraging"
            self.foraging_counter += 1
            direction = plant_close.pos - self.pos
            norm = np.linalg.norm(direction) + 1e-5
            self.vel = (direction / norm) * self.base_speed * speed_global
            self.target = plant_close
        else:
            threats = [g for g in goblins if distance(self.pos, g.pos) < 100]
            if threats:
                self.state = "fleeing"
                avg_threat = np.mean([g.pos for g in threats], axis=0)
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

        # On-contact foraging: if close to a plant, forage it.
        collision_threshold = 10
        for p in plants:
            if p.active and distance(self.pos, p.pos) < collision_threshold:
                self.energy += 10
                self.hunger_timer = 900
                p.active = False
                p.regrow_timer = PLANT_REGROW_TIME

        # On-contact damage: if in contact with a goblin, damage it.
        for g in goblins[:]:
            if distance(self.pos, g.pos) < collision_threshold:
                g.health -= self.attack_damage
                if g.health <= 0:
                    self.reproduction_weight += 0.5
                # (Damage occurs on contact; you can break here if you wish only one collision per frame.)
                break

        # Apply terrain effects.
        self.vel *= get_terrain_factor(self.pos)
        self.update_position()

        # Resolve collisions with all nearby objects (goblins and plants).
        all_objects = []
        all_objects.extend(goblins)
        all_objects.extend(plants)
        resolve_collisions(self, all_objects, 10)

# -------------------------------
# Scavenger Class (Giant Spiders - Very Solitary)
# -------------------------------
class Scavenger(Creature):
    def __init__(self, x: float, y: float, species: str = None, action_weights: dict = None):
        """
        Initialize a Scavenger creature.
        """
        super().__init__(x, y, 40, base_speed=5.0, hunger_duration=750)
        self.energy = 0
        self.steal_amount = 20
        self.species = species if species is not None else random.choice(["spiderA", "spiderB"])
        self.action_weights = action_weights if action_weights is not None else {"steal": 1.0, "forage": 1.0}
        self.attack_damage = 3
        # Two-layer network parameters for processing movement when stealing.
        self.weights1 = np.random.randn(2, 4)
        self.bias1 = np.random.randn(4)
        self.weights2 = np.random.randn(4, 2)
        self.bias2 = np.random.randn(2)
        self.reproduction_cooldown = 100

    def update(self, goblins: list, plants: list, scavengers: list, speed_global: float = 1.0):
        # Decrement reproduction cooldown.
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1

        # Decide action: if any goblin with sufficient energy is nearby, choose "steal", else "forage".
        action = "forage"
        targets = [g for g in goblins if g.energy > 20 and distance(self.pos, g.pos) < 150]
        if targets:
            total = self.action_weights["steal"] + self.action_weights["forage"]
            if random.random() < (self.action_weights["steal"] / total):
                action = "steal"

        if action == "steal":
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
                move_direction = direction / norm
                hidden = np.tanh(np.dot(move_direction, self.weights1) + self.bias1)
                output = np.tanh(np.dot(hidden, self.weights2) + self.bias2)
                self.vel = output * self.base_speed * speed_global
                self.target = target
            else:
                self.vel = np.array([0.0, 0.0])
                self.target = None
        else:  # forage
            target = None
            min_d = float('inf')
            for p in plants:
                if p.active:
                    d = distance(self.pos, p.pos)
                    if d < min_d:
                        min_d = d
                        target = p
            if target is not None:
                direction = target.pos - self.pos
                norm = np.linalg.norm(direction) + 1e-5
                self.vel = (direction / norm) * self.base_speed * speed_global
                self.target = target
            else:
                self.vel = np.array([0.0, 0.0])
                self.target = None

        # On-contact stealing: if any goblin is within collision_threshold, steal energy.
        collision_threshold = 10
        for g in goblins:
            if g.energy > 0 and distance(self.pos, g.pos) < collision_threshold:
                stolen = min(self.steal_amount, g.energy)
                g.energy -= stolen
                self.energy += stolen
                self.action_weights["steal"] *= 1.1
                break

        # On-contact foraging: if within collision_threshold of an active plant, forage it.
        for p in plants:
            if p.active and distance(self.pos, p.pos) < collision_threshold:
                self.hunger_timer = 750
                p.active = False
                p.regrow_timer = PLANT_REGROW_TIME
                break

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
            separation /= count
            self.vel += separation * 2.0

        # Apply terrain effects.
        self.vel *= get_terrain_factor(self.pos)
        self.update_position()

        # Resolve collisions with all nearby objects (goblins and plants).
        all_objects = []
        all_objects.extend(goblins)
        all_objects.extend(plants)
        resolve_collisions(self, all_objects, 10)

# -------------------------------
# Predator Class (Wolves) – Now also target scavengers
# Also predators maintain solitary spacing.
# -------------------------------
class Predator(Creature):
    def __init__(self, x: float, y: float, species: str = None):
        """
        Initialize a Predator instance.
        """
        super().__init__(x, y, 80, base_speed=7.0, hunger_duration=800)
        self.energy = 0.0
        self.species = species if species is not None else random.choice(["wolfA", "wolfB"])
        self.attack_damage = 20
        self.attack_cooldown = 10
        self.reproduction_cooldown = 100
        # Additional state attributes.
        self.state = "active"  # "hunting", "patrolling", or "resting"
        self.rest_threshold = 20
        self.patrolling_center = np.array([WIDTH / 2, HEIGHT / 2])
        self.solitary_distance = 100

    def compute_solitary_force(self, predators: list) -> np.ndarray:
        force = np.array([0.0, 0.0])
        count = 0
        for other in predators:
            if other is not self:
                d = distance(self.pos, other.pos)
                if d < self.solitary_distance:
                    force += (self.pos - other.pos) / (d + 1e-5)
                    count += 1
        if count > 0:
            return force / count
        return force

    def update_state(self, prey_list: list) -> None:
        if self.energy < self.rest_threshold:
            self.state = "resting"
        elif prey_list:
            self.state = "hunting"
        else:
            self.state = "patrolling"

    def update(self, goblins: list, boars: list, scavengers: list, predators: list,
               speed_global: float = 1.0):
        # Decrement cooldowns.
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1

        # Build prey list.
        prey_list = goblins + boars + scavengers
        self.update_state(prey_list)

        if self.state == "resting":
            self.base_speed = 3.0
            self.energy += 0.5
            rand_dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
            norm = np.linalg.norm(rand_dir) + 1e-5
            self.vel = (rand_dir / norm) * self.base_speed * speed_global
        elif self.state == "hunting" and prey_list:
            target = None
            min_d = float('inf')
            goblin_weight = 1.5  # Make goblins less attractive
            for p in prey_list:
                d = distance(self.pos, p.pos)
                if isinstance(p, Goblin):
                    d *= goblin_weight
                if d < min_d:
                    min_d = d
                    target = p
            if target:
                direction = target.pos - self.pos
                norm = np.linalg.norm(direction) + 1e-5
                self.vel = (direction / norm) * self.base_speed * speed_global
                self.target = target
            else:
                rand_dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                norm = np.linalg.norm(rand_dir) + 1e-5
                self.vel = (rand_dir / norm) * self.base_speed * speed_global
        elif self.state == "patrolling":
            direction = self.patrolling_center - self.pos
            norm = np.linalg.norm(direction) + 1e-5
            self.vel = (direction / norm) * self.base_speed * speed_global

        if self.state != "resting":
            self.base_speed = 7.0

        # On-contact attack: if in contact with any prey, damage them.
        collision_threshold = 10
        for p in prey_list[:]:
            if distance(self.pos, p.pos) < collision_threshold:
                p.health -= self.attack_damage
                if p.health <= 0:
                    self.energy += 50
                    # Remove the prey from its list.
                    if p in goblins:
                        goblins.remove(p)
                    elif p in boars:
                        boars.remove(p)
                    elif p in scavengers:
                        scavengers.remove(p)
                self.attack_cooldown = 10
                break

        # Apply solitary force.
        solitary_force = self.compute_solitary_force(predators) * 2.0
        self.vel += solitary_force

        # Apply terrain effects.
        self.vel *= get_terrain_factor(self.pos)
        self.update_position()

        # Resolve collisions with all creatures (goblins, boars, scavengers) and plants.
        all_objects = []
        all_objects.extend(goblins)
        all_objects.extend(boars)
        all_objects.extend(scavengers)
        # Optionally, include predators if desired.
        resolve_collisions(self, all_objects, 10)

    def try_attack(self, goblins: list, boars: list, scavengers: list) -> None:
        # On-contact attack is handled in update().
        pass

    def try_reproduce(self, predator_list: list) -> None:
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
                    self.reproduction_cooldown = 100
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
            "q_weights1": [[0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1]],
            "q_bias1": [0.1, 0.1, 0.1, 0.1],
            "q_weights2": [[0.1, 0.1],
                           [0.1, 0.1],
                           [0.1, 0.1],
                           [0.1, 0.1]],
            "q_bias2": [0.1, 0.1]
        }
        with open(q_params_file, "w") as f:
            json.dump(default_params, f)

    # Initialize populations (adjusted for larger screen density)
    goblins = [Goblin(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(18)]
    for g in goblins:
        g.load_q_params(q_params_file)
    boars = [Boar(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(27)]
    scavengers = [Scavenger(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(18)]
    predators = [Predator(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(2)]
    plants = [Plant(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(35)]

    # Global territory map for goblins and cover points for ambush behavior.
    global_territory_map = {}
    terrain_points = [np.array([zone.rect.centerx, zone.rect.centery]) for zone in TERRAIN_ZONES]

    frame_count = 0
    speed_factor = 8.0  # Global speed factor
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0  # Delta time in seconds
        frame_count += 1

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Update Resources ---
        for plant in plants:
            plant.update(all_plants=plants)
        # Spawn additional plants if needed.
        if len(plants) < 50 and random.random() < 0.02:
            plants.append(Plant(random.randint(0, WIDTH), random.randint(0, HEIGHT)))

        # --- Update Creatures ---
        # Update Goblins (pass extra parameters for terrain and territory)
        for g in goblins[:]:
            g.update(boars, goblins, plants, predators,
                    speed_global=speed_factor * dt,
                    nearby_terrain=terrain_points,
                    territory_map=global_territory_map)
            g.try_forage(plants, boars, predators)  # Add this line
            g.try_attack(boars, predators, plants)
            g.try_reproduce(goblins)
            if g.is_dead() or g.update_hunger():
                goblins.remove(g)

        # Update Boars (their update now handles on-contact damage and foraging)
        for b in boars[:]:
            b.update(goblins, plants, speed_global=speed_factor * dt)
            if b.is_dead() or b.update_hunger():
                boars.remove(b)

        # Update Scavengers
        for s in scavengers[:]:
            s.update(goblins, plants, scavengers, speed_global=speed_factor * dt)
            if s.is_dead() or s.update_hunger():
                scavengers.remove(s)

        # Update Predators
        for p in predators[:]:
            p.update(goblins, boars, scavengers, predators, speed_global=speed_factor * dt)
            if p.is_dead() or p.update_hunger():
                predators.remove(p)

        # --- External Spawning ---
        # Spawn new scavengers if population is low; rate based on goblin count.
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
        # Spawn new boars if population is low; rate based on plant count.
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
        # Spawn new predators if population is low; rate based on combined boar and goblin counts.
        if len(predators) < 2 and random.random() < ((len(boars) + len(goblins)) / 200.0):
            edge = random.randint(0, 2)
            if edge == 0:
                x, y = -10, random.randint(0, HEIGHT)
            elif edge == 1:
                x, y = WIDTH + 10, random.randint(0, HEIGHT)
            else:
                x, y = random.randint(0, WIDTH), -10
            predators.append(Predator(x, y))

        # --- Drawing ---
        screen.fill((50, 50, 80))  # Background color
        # Draw terrain zones.
        for zone in TERRAIN_ZONES:
            pygame.draw.rect(screen, (30, 60, 30), zone.rect, 2)
        # Draw plants.
        for plant in plants:
            if plant.active:
                pygame.draw.circle(screen, (0, 255, 0),
                                   (int(plant.pos[0]), int(plant.pos[1])), 5)
        # Draw goblins with velocity lines.
        for g in goblins:
            pygame.draw.circle(screen, g.color,
                               (int(g.pos[0]), int(g.pos[1])), 5)
            end = (int(g.pos[0] + g.vel[0] * 5), int(g.pos[1] + g.vel[1] * 5))
            pygame.draw.line(screen, (0, 0, 0),
                             (int(g.pos[0]), int(g.pos[1])), end, 2)
        # Draw boars (change color if foraging).
        for b in boars:
            col = (139, 69, 19) if b.state != "foraging" else (0, 0, 255)
            pygame.draw.circle(screen, col,
                               (int(b.pos[0]), int(b.pos[1])), 8)
        # Draw scavengers.
        for s in scavengers:
            pygame.draw.circle(screen, (128, 0, 128),
                               (int(s.pos[0]), int(s.pos[1])), 6)
        # Draw predators.
        for p in predators:
            pygame.draw.circle(screen, (100, 100, 100),
                               (int(p.pos[0]), int(p.pos[1])), 7)

        # Display simulation metrics.
        metrics = (f"Goblins: {len(goblins)}  Boars: {len(boars)}  "
                   f"Scavengers: {len(scavengers)}  Predators: {len(predators)}  "
                   f"Plants: {len(plants)}")
        text_surface = font.render(metrics, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        pygame.display.flip()

        # --- Logging ---
        if frame_count % 30 == 0:
            log_line = f"{time.time()},{len(goblins)},{len(boars)},{len(scavengers)},{len(predators)},{len(plants)}\n"
            log_file.write(log_line)
            log_file.flush()

        # --- Restart Goblins if Extinct ---
        if len(goblins) == 0:
            # Save Q-learning parameters (could be improved by saving best performing parameters)
            if goblins:
                goblins[0].save_q_params(q_params_file)
            goblins = [Goblin(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(10)]
            for g in goblins:
                g.load_q_params(q_params_file)

    log_file.close()
    pygame.quit()


if __name__ == "__main__":
    main()