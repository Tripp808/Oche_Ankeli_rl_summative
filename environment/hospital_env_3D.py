import pygame
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import imageio
import sys
import random
import math
from datetime import datetime
import os
from PIL import Image
import gymnasium as gym
from gymnasium import spaces

class HospitalSimulation3D(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode='human', width=800, height=600, max_steps=float('inf'), mode='delivery',
                 auto_movement=True):
        super(HospitalSimulation3D, self).__init__()
        
        # Gym environment settings
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(14,), dtype=np.float32
        )
        
        # Set up dimensions and display
        self.width = width
        self.height = height
        self.display = (width, height)
        self.render_mode = render_mode
        self.max_steps = max_steps  # infinity by default
        self.step_count = 0
        self.cumulative_reward = 0
        self.reward = 0
        self.mode = mode  # 'delivery' or 'emergency'
        
        # Real-time reward visualization
        self.reward_history = []  # Store recent rewards for visualization
        self.reward_particles = []  # Particles for visual reward effects
        self.reward_text_timeout = 0  # Timer for reward text display
        self.reward_text = ""  # Current reward text to display
        self.reward_color = (0, 255, 0)  # Color for reward text (green for positive)
        
        # Automatic movement settings
        self.auto_movement = auto_movement
        self.auto_path = []
        self.current_path_index = 0
        self.movement_timer = 0
        self.movement_delay = 10  # Frames between movements
        
        # Camera settings
        self.camera_modes = ["follow", "overview", "first_person", "side_view"]
        self.current_camera_mode = 0  # Start with follow camera
        self.camera_switch_timer = 0
        self.camera_switch_delay = 300  # Frames between automatic camera switches
        self.camera_height = 6
        self.camera_distance = 5
        
        # Current date and user information (for display purposes)
        self.current_date = "2025-04-03 15:04:10"  # Using the updated date
        self.current_user = "Emmanuel-Begati"  # Using the provided username
        
        # Define enhanced colors for 3D objects - more vibrant and distinctive
        self.colors_3d = {
            'floor': (0.8, 0.8, 0.8),
            'wall': (0.95, 0.95, 0.95),
            'bed': (0.9, 0.9, 1.0),
            'bed_frame': (0.6, 0.6, 0.7),
            'delivery_room': (0.7, 1.0, 0.7, 0.4),  # More vibrant green
            'prenatal_care': (0.7, 0.7, 1.0, 0.4),  # More vibrant blue
            'emergency_room': (1.0, 0.7, 0.7, 0.4),  # More vibrant red
            'crowded_area': (1.0, 0.9, 0.7, 0.4),  # More vibrant yellow
            'skin': (0.9, 0.75, 0.65),
            'dress': (0.3, 0.6, 1.0),  # More vibrant blue dress
            'belly': (0.95, 0.8, 0.75),
            'reception': (0.7, 0.5, 0.3),
            'doctor': (0.95, 0.2, 0.2),  # Brighter red
            'nurse': (0.2, 0.6, 1.0),  # Brighter blue
            'medical_equipment': (0.5, 0.5, 0.5)
        }
        
        # Define colors for 2D UI elements with improved contrast
        self.colors = {
            'wall': (100, 100, 100),        # Gray
            'floor': (240, 240, 240),       # Light gray
            'emergency': (255, 150, 150),   # Brighter red for emergency room
            'delivery': (150, 255, 150),    # Brighter green for delivery room
            'reception': (255, 235, 205),   # Light brown/beige
            'hallway': (200, 200, 255),     # Brighter lavender
            'door': (160, 82, 45),          # Brown door
            'agent': (50, 100, 255),        # Brighter blue for agent
            'nurse': (255, 105, 180),       # Pink
            'bed': (200, 200, 255),         # Light blue
            'target': (255, 255, 0),        # Yellow
            'positive_reward': (50, 200, 50),  # Green for positive rewards
            'negative_reward': (200, 50, 50),  # Red for negative rewards
            'neutral_reward': (200, 200, 50)   # Yellow for neutral rewards
        }
        
        # Position indicators for key locations
        self.position_indicators = {
            'delivery_room': {'active': True, 'color': (0, 255, 0, 180), 'height': 0.5},
            'emergency_room': {'active': True, 'color': (255, 0, 0, 180), 'height': 0.5},
            'reception': {'active': True, 'color': (200, 150, 50, 180), 'height': 0.3},
            'prenatal_care': {'active': True, 'color': (50, 50, 255, 180), 'height': 0.3}
        }
        
        # Room labels and indicators
        self.room_labels = {
            'emergency_room': "Emergency Room",
            'delivery_room': "Delivery Room",
            'reception': "Reception",
            'prenatal_care': "Prenatal Care",
            'entrance': "Entrance",
            'hallway_center': "Hallway"
        }
        
        # Initialize important locations
        self.locations = {
            'delivery_room': {'pos': [7.0, 7.0, 0], 'size': [3, 3]},
            'prenatal_care': {'pos': [3.0, 8.0, 0], 'size': [2, 2]},
            'emergency_room': {'pos': [8.0, 2.0, 0], 'size': [2, 2]},
            'reception': {'pos': [2.0, 8.0, 0], 'size': [2, 1]},
            'entrance': {'pos': [2.0, 2.0, 0], 'size': [1, 1]},
            'hallway_center': {'pos': [5.0, 5.0, 0], 'size': [4, 4]}
        }
        
        # Initialize agent properties (near entrance)
        self.agent_pos = self.locations['entrance']['pos'].copy()
        self.agent_rotation = 0
        self.agent_speed = 0.06
        
        # Set target based on mode
        if self.mode == 'delivery':
            self.target_pos = [7.0, 7.0, 0]  # Delivery room center
            self.target_region = self.locations['delivery_room']
        else:  # emergency
            self.target_pos = [8.0, 2.0, 0]  # Emergency room center
            self.target_region = self.locations['emergency_room']
            
        # Initialize crowded areas (as obstacles)
        self.crowded_areas = [
            {'pos': [4.0, 5.0, 0], 'radius': 1.5, 'density': 0.8},  # Waiting area
            {'pos': [5.0, 2.0, 0], 'radius': 1.0, 'density': 0.6}   # Registration
        ]
        
        # Initialize medical staff
        self.medical_staff = []
        for i in range(3):  # 2 doctors and 1 nurse
            staff_type = 'doctor' if i < 2 else 'nurse'
            self.medical_staff.append({
                'type': staff_type,
                'pos': [random.uniform(1, 9), random.uniform(1, 9), 0],
                'rotation': random.uniform(0, 360),
                'speed': random.uniform(0.02, 0.04),
                'direction': random.uniform(0, 2*math.pi)
            })
            
        # Set hospital beds
        self.beds = [
            [2, 3, 0],  # Position of bed 1
            [6, 3, 0],  # Position of bed 2
            [6, 7, 0]   # Position of bed 3
        ]
        
        # Create output directory if it doesn't exist
        self.output_dir = "simulation_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize pygame and OpenGL if rendering
        if self.render_mode is not None:
            pygame.init()
            pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
            pygame.display.set_caption("MamaSafe Navigator - 3D Hospital Simulation")
            
            # Setup OpenGL
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            
            # Configure lighting
            light_position = [10.0, 10.0, 10.0, 1.0]
            light_color = [1.0, 1.0, 1.0, 1.0]
            glLightfv(GL_LIGHT0, GL_POSITION, light_position)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, light_color)
            
            # Setup perspective
            glMatrixMode(GL_PROJECTION)
            gluPerspective(45, (width/height), 0.1, 50.0)
            glMatrixMode(GL_MODELVIEW)
            
            # Initialize pygame font
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 18)
            self.small_font = pygame.font.SysFont('Arial', 14)
            self.large_font = pygame.font.SysFont('Arial', 24, bold=True)
            self.clock = pygame.time.Clock()
            
            # Create display lists for efficiency
            self.hospital_list = self.create_hospital_display_list()
            self.agent_list = self.create_agent_display_list()
            self.staff_list = self.create_staff_display_list()
            
        # Generate an automatic tour path
        self._generate_auto_path()
    
    def _generate_auto_path(self):
        """Generate an automatic tour path through the hospital"""
        # Define key points to visit
        key_locations = [
            self.locations['entrance']['pos'][:2],
            self.locations['hallway_center']['pos'][:2],
            self.locations['emergency_room']['pos'][:2],
            self.locations['hallway_center']['pos'][:2],
            self.locations['prenatal_care']['pos'][:2],
            self.locations['hallway_center']['pos'][:2],
            self.locations['reception']['pos'][:2],
            self.locations['hallway_center']['pos'][:2],
            self.locations['delivery_room']['pos'][:2],
            self.locations['hallway_center']['pos'][:2],
            self.locations['entrance']['pos'][:2]
        ]
        
        # Create path with intermediate points
        path = []
        for i in range(len(key_locations) - 1):
            start = key_locations[i]
            end = key_locations[i+1]
            
            # Calculate distance
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Add intermediate points based on distance
            steps = max(2, int(dist * 2))
            for step in range(steps + 1):
                t = step / steps
                x = start[0] + t * dx
                y = start[1] + t * dy
                path.append([x, y, 0])
        
        self.auto_path = path
    
    def _add_reward_particles(self, value):
        """Add visual particles for reward feedback"""
        if abs(value) < 0.01:  # Ignore very small rewards
            return
            
        # Determine particle count and color based on reward value
        if value > 0:
            count = min(20, int(value * 2))
            color = self.colors['positive_reward']
            text = f"+{value:.1f}"
        else:
            count = min(20, int(abs(value) * 2))
            color = self.colors['negative_reward']
            text = f"{value:.1f}"
            
        # Set reward text for display
        self.reward_text = text
        self.reward_color = color
        self.reward_text_timeout = 60  # Show for 60 frames
        
        # Create particles around agent
        agent_x, agent_y = self.agent_pos[0], self.agent_pos[1]
        
        for _ in range(count):
            # Random position around agent
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0.5, 2.0)
            x = agent_x + math.cos(angle) * distance
            y = agent_y + math.sin(angle) * distance
            z = random.uniform(0.5, 2.0)
            
            # Random velocity
            vx = random.uniform(-0.03, 0.03)
            vy = random.uniform(-0.03, 0.03)
            vz = random.uniform(0.02, 0.08)
            
            # Random size and lifetime
            size = random.uniform(0.05, 0.15)
            lifetime = random.randint(20, 40)
            
            # Add particle
            self.reward_particles.append({
                'pos': [x, y, z],
                'vel': [vx, vy, vz],
                'color': color,
                'size': size,
                'age': 0,
                'lifetime': lifetime
            })
    
    def _update_reward_particles(self):
        """Update and age reward particles"""
        # Update existing particles
        for particle in self.reward_particles[:]:
            # Update position
            particle['pos'][0] += particle['vel'][0]
            particle['pos'][1] += particle['vel'][1]
            particle['pos'][2] += particle['vel'][2]
            
            # Apply gravity and friction
            particle['vel'][2] -= 0.005  # Gravity
            particle['vel'][0] *= 0.98   # Friction
            particle['vel'][1] *= 0.98   # Friction
            
            # Age particle
            particle['age'] += 1
            
            # Remove if too old
            if particle['age'] >= particle['lifetime']:
                self.reward_particles.remove(particle)
        
        # Update reward text timeout
        if self.reward_text_timeout > 0:
            self.reward_text_timeout -= 1
    
    def _update_auto_movement(self):
        """Update agent position when auto-movement is enabled"""
        if not self.auto_path:
            return False
            
        self.movement_timer += 1
        if self.movement_timer < self.movement_delay:
            return False
            
        self.movement_timer = 0
        
        # Get current and next waypoint
        current_index = self.current_path_index
        next_index = (current_index + 1) % len(self.auto_path)
        
        # Move to next waypoint
        next_pos = self.auto_path[next_index]
        self.agent_pos = next_pos
        
        # Update agent rotation to face movement direction
        if next_index > current_index or (current_index == len(self.auto_path) - 1 and next_index == 0):
            dx = self.auto_path[next_index][0] - self.auto_path[current_index][0]
            dy = self.auto_path[next_index][1] - self.auto_path[current_index][1]
            
            # Calculate angle in degrees
            angle = math.degrees(math.atan2(dy, dx))
            
            # Convert to the 0-360 rotation system
            self.agent_rotation = (90 - angle) % 360
        
        # Update path index
        self.current_path_index = next_index
        
        return True
    
    def _update_camera_view(self):
        """Update camera view mode periodically"""
        self.camera_switch_timer += 1
        if self.camera_switch_timer >= self.camera_switch_delay:
            self.camera_switch_timer = 0
            self.current_camera_mode = (self.current_camera_mode + 1) % len(self.camera_modes)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset step counter and reward
        self.step_count = 0
        self.cumulative_reward = 0
        self.reward = 0
        
        # Reset reward visualization
        self.reward_particles = []
        self.reward_text_timeout = 0
        self.reward_history = []
        
        # Reset agent position to entrance
        self.agent_pos = self.locations['entrance']['pos'].copy()
        self.agent_rotation = 0
        
        # Reset medical staff positions
        for i, staff in enumerate(self.medical_staff):
            staff['pos'] = [random.uniform(1, 9), random.uniform(1, 9), 0]
            staff['rotation'] = random.uniform(0, 360)
            staff['direction'] = random.uniform(0, 2*math.pi)
        
        # Reset automatic movement
        self.current_path_index = 0
        self.movement_timer = 0
        self.camera_switch_timer = 0
        
        # Return initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        # Increment step counter
        self.step_count += 1
        
        # Default reward is small negative to encourage efficient paths
        prev_reward = self.reward
        self.reward = -0.01
        
        # If automatic movement is enabled, override the action with auto-movement
        if self.auto_movement:
            self._update_auto_movement()
            self._update_camera_view()
        else:
            # Calculate new position based on action
            new_pos = self.agent_pos.copy()
            if action == 0:  # Up (North)
                new_pos[1] += self.agent_speed * 2
                self.agent_rotation = 0
            elif action == 1:  # Right (East)
                new_pos[0] += self.agent_speed * 2
                self.agent_rotation = 90
            elif action == 2:  # Down (South)
                new_pos[1] -= self.agent_speed * 2
                self.agent_rotation = 180
            elif action == 3:  # Left (West)
                new_pos[0] -= self.agent_speed * 2
                self.agent_rotation = 270
            
            # Check if the new position is valid (within bounds and not colliding)
            collision = self._check_collision(new_pos)
            
            if not collision:
                self.agent_pos = new_pos
            else:
                # Penalty for collisions
                self.reward -= 0.1
        
        # Check if the agent reached a special location
        location_reached = self._check_location_reached()
        
        # Award location-specific rewards
        if location_reached:
            if location_reached == 'delivery_room' and self.mode == 'delivery':
                self.reward += 50.0
            elif location_reached == 'emergency_room' and self.mode == 'emergency':
                self.reward += 50.0
            elif location_reached == 'prenatal_care':
                self.reward += 10.0
            elif location_reached == 'reception':
                self.reward += 5.0
        
        # Move medical staff
        if self.step_count % 3 == 0:  # Move staff every 3 steps
            self._update_staff_positions()
        
        # Check for collisions with staff or crowded areas
        collision_type = self._check_special_collision()
        if collision_type == 'staff':
            self.reward -= 10.0
        elif collision_type == 'crowded':
            self.reward -= 5.0
        
        # Update cumulative reward
        self.cumulative_reward += self.reward
        
        # Store reward in history (for reward graph)
        self.reward_history.append(self.reward)
        if len(self.reward_history) > 100:  # Keep only the last 100 rewards
            self.reward_history.pop(0)
        
        # Add visual reward particles if reward changed significantly
        if abs(self.reward - prev_reward) > 0.1:
            self._add_reward_particles(self.reward)
        
        # Update reward particles
        self._update_reward_particles()
        
        # Check if task is complete (reached target based on mode)
        in_target_region = self._is_in_target_region()
        
        # Determine termination conditions - now only terminates if explicitly set
        # The environment will keep running indefinitely for visualization
        terminated = False  # Removed termination conditions to run indefinitely
        
        # Bonus reward for reaching target
        if in_target_region:
            self.reward += 100
            self.cumulative_reward += 100
            self._add_reward_particles(100)  # Add particles for big reward
        
        # Get observation
        observation = self._get_observation()
        
        # Return step information
        info = {
            'step_count': self.step_count,
            'location': location_reached,
            'collision': collision_type
        }
        truncated = False
        
        return observation, self.reward, terminated, truncated, info
    
    def _check_collision(self, pos):
        """Check if position is valid (inside bounds and not in walls)"""
        # Check if out of bounds
        if pos[0] < 0.5 or pos[0] > 9.5 or pos[1] < 0.5 or pos[1] > 9.5:
            return True
        
        # Check for collisions with walls or other obstacles (simplified)
        # In a real implementation, this would check against actual wall coordinates
        
        return False
    
    def _check_special_collision(self):
        """Check for collisions with staff or crowded areas"""
        # Check staff collisions
        for staff in self.medical_staff:
            dist = math.sqrt((self.agent_pos[0] - staff['pos'][0])**2 + 
                           (self.agent_pos[1] - staff['pos'][1])**2)
            if dist < 0.5:  # Collision radius
                return 'staff'
        
        # Check crowded areas
        for area in self.crowded_areas:
            dist = math.sqrt((self.agent_pos[0] - area['pos'][0])**2 + 
                           (self.agent_pos[1] - area['pos'][1])**2)
            if dist < area['radius'] * 0.7:  # Only count as collision if deep in the crowd
                return 'crowded'
        
        return None
    
    def _check_location_reached(self):
        """Check if agent has reached an important location"""
        for loc_name, loc_data in self.locations.items():
            # Skip the hallway center, it's not a real room
            if loc_name == 'hallway_center':
                continue
                
            pos = loc_data['pos']
            size = loc_data['size']
            
            # Check if agent is within the location boundaries
            if (pos[0] - size[0]/2 <= self.agent_pos[0] <= pos[0] + size[0]/2 and
                pos[1] - size[1]/2 <= self.agent_pos[1] <= pos[1] + size[1]/2):
                return loc_name
        
        return None
    
    def _is_in_target_region(self):
        """Check if the agent is in the target region based on mode"""
        if self.mode == 'delivery':
            target = self.locations['delivery_room']
        else:
            target = self.locations['emergency_room']
        
        pos = target['pos']
        size = target['size']
        
        # Check if agent is well within the target area
        return (pos[0] - size[0]/3 <= self.agent_pos[0] <= pos[0] + size[0]/3 and
                pos[1] - size[1]/3 <= self.agent_pos[1] <= pos[1] + size[1]/3)
    
    def _update_staff_positions(self):
        """Update positions of medical staff with random movement"""
        for staff in self.medical_staff:
            # Random direction changes
            if random.random() < 0.02:
                staff['direction'] += random.uniform(-0.5, 0.5)
            
            # Move staff
            staff['pos'][0] += math.cos(staff['direction']) * staff['speed']
            staff['pos'][1] += math.sin(staff['direction']) * staff['speed']
            
            # Bounce off walls
            if staff['pos'][0] < 1 or staff['pos'][0] > 9:
                staff['direction'] = math.pi - staff['direction']
            if staff['pos'][1] < 1 or staff['pos'][1] > 9:
                staff['direction'] = -staff['direction']
            
            # Keep within bounds
            staff['pos'][0] = np.clip(staff['pos'][0], 1, 9)
            staff['pos'][1] = np.clip(staff['pos'][1], 1, 9)
            
            # Update rotation for display
            staff['rotation'] = math.degrees(staff['direction'])
    
    def _get_observation(self):
        """Return the observation vector required by the agent"""
        # Agent position (2)
        agent_x, agent_y = self.agent_pos[0], self.agent_pos[1]
        
        # Target position (2)
        target_x, target_y = self.target_pos[0], self.target_pos[1]
        
        # Walls around agent (4) - simplified version
        # In a real implementation, this would check actual wall positions
        wall_up = 1 if agent_y > 9 else 0
        wall_right = 1 if agent_x > 9 else 0
        wall_down = 1 if agent_y < 1 else 0
        wall_left = 1 if agent_x < 1 else 0
        walls_around = [wall_up, wall_right, wall_down, wall_left]
        
        # Get relative positions of the 3 closest staff (6)
        staff_relative_positions = []
        staff_distances = []
        
        for i, staff in enumerate(self.medical_staff):
            dx = staff['pos'][0] - agent_x
            dy = staff['pos'][1] - agent_y
            dist = math.sqrt(dx**2 + dy**2)
            staff_distances.append((dist, dx, dy, i))
        
        # Sort by distance
        staff_distances.sort()
        
        # Add relative positions of up to 3 closest staff
        for i in range(min(3, len(staff_distances))):
            dist, dx, dy, _ = staff_distances[i]
            staff_relative_positions.extend([dx, dy])
        
        # Pad with zeros if less than 3 staff
        while len(staff_relative_positions) < 6:
            staff_relative_positions.extend([0, 0])
        
        # Combine all observation components
        observation = np.array(
            [agent_x, agent_y] + [target_x, target_y] + walls_around + staff_relative_positions,
            dtype=np.float32
        )
        
        return observation
    
    def _calculate_distance_to_target(self):
        """Calculate Euclidean distance from agent to target"""
        agent_x, agent_y = self.agent_pos[0], self.agent_pos[1]
        target_x, target_y = self.target_pos[0], self.target_pos[1]
        return math.sqrt((target_x - agent_x)**2 + (target_y - agent_y)**2)
    
    def render(self):
        if self.render_mode is None:
            return
        
        # Clear buffers
        glClearColor(0.7, 0.7, 0.7, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Set camera based on current mode
        self._set_camera_view()
        
        # Draw hospital environment
        glCallList(self.hospital_list)
        
        # Draw position indicators for rooms
        self._render_position_indicators()
        
        # Draw reward particles in 3D
        self._render_reward_particles()
        
        # Draw medical staff
        for staff in self.medical_staff:
            glPushMatrix()
            glTranslatef(staff['pos'][0], staff['pos'][1], staff['pos'][2])
            glRotatef(staff['rotation'], 0, 0, 1)
            glCallList(self.staff_list[staff['type']])
            glPopMatrix()
        
        # Draw the agent
        glPushMatrix()
        glTranslatef(self.agent_pos[0], self.agent_pos[1], self.agent_pos[2])
        glRotatef(self.agent_rotation, 0, 0, 1)
        glCallList(self.agent_list)
        glPopMatrix()
        
        # Check for special conditions for UI display
        collision_type = self._check_special_collision()
        location_reached = self._check_location_reached()
        
        # Render UI overlay with step counter, rewards, etc.
        self._render_ui_overlay(collision_type, location_reached)
        
        # Render the minimap
        self._render_minimap()
        
        # Render the color key
        self._render_color_key()
        
        # Render the directional guidance to target
        self._render_directional_guidance()
        
        # Render camera mode indicator
        self._render_camera_mode_indicator()
        
        # Render real-time reward
        self._render_reward_indicator()
        
        # Update display
        pygame.display.flip()
        
        # Capture frame for recording if needed
        if self.render_mode == 'rgb_array':
            data = pygame.image.tostring(pygame.display.get_surface(), 'RGB')
            surface = pygame.image.fromstring(data, self.display, 'RGB')
            frame = pygame.surfarray.array3d(surface)
            return np.swapaxes(frame, 0, 1)
        
        # Control frame rate
        self.clock.tick(self.metadata["render_fps"])
        
        return None
    
    def _render_reward_particles(self):
        """Render reward particles in 3D space"""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        for particle in self.reward_particles:
            # Calculate alpha based on age
            alpha = 1.0 - (particle['age'] / particle['lifetime'])
            
            # Set color with appropriate alpha
            color = particle['color']
            glColor4f(color[0]/255, color[1]/255, color[2]/255, alpha)
            
            # Draw particle as a small cube
            glPushMatrix()
            glTranslatef(particle['pos'][0], particle['pos'][1], particle['pos'][2])
            size = particle['size'] * (1.0 - 0.5 * (particle['age'] / particle['lifetime']))
            self.draw_cube(0, 0, 0, size, size, size)
            glPopMatrix()
        
        glDisable(GL_BLEND)
    
    def _render_position_indicators(self):
        """Render 3D indicators for important locations"""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Pulse effect for indicators
        pulse = 0.2 * math.sin(self.step_count * 0.1) + 0.8
        
        # Draw indicators for each room
        for room_name, indicator in self.position_indicators.items():
            if not indicator['active'] or room_name not in self.locations:
                continue
                
            room_data = self.locations[room_name]
            pos = room_data['pos']
            
            # Set color with pulse effect
            color = indicator['color']
            glColor4f(color[0]/255, color[1]/255, color[2]/255, color[3]/255 * pulse)
            
            # Draw pillar of light
            height = indicator['height'] * 4 * pulse
            width = 0.3 + 0.1 * pulse
            
            # Base of pillar
            glPushMatrix()
            glTranslatef(pos[0], pos[1], 0.05)  # Slightly above floor
            
            # Draw vertical beam
            glBegin(GL_QUADS)
            glVertex3f(-width/2, -width/2, 0)
            glVertex3f(width/2, -width/2, 0)
            glVertex3f(width/2, -width/2, height)
            glVertex3f(-width/2, -width/2, height)
            
            glVertex3f(-width/2, width/2, 0)
            glVertex3f(width/2, width/2, 0)
            glVertex3f(width/2, width/2, height)
            glVertex3f(-width/2, width/2, height)
            
            glVertex3f(-width/2, -width/2, 0)
            glVertex3f(-width/2, width/2, 0)
            glVertex3f(-width/2, width/2, height)
            glVertex3f(-width/2, -width/2, height)
            
            glVertex3f(width/2, -width/2, 0)
            glVertex3f(width/2, width/2, 0)
            glVertex3f(width/2, width/2, height)
            glVertex3f(width/2, -width/2, height)
            glEnd()
            
            glPopMatrix()
        
        glDisable(GL_BLEND)
    
    def _render_reward_indicator(self):
        """Render reward indicator showing current reward value"""
        # Switch to 2D projection
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, self.height, 0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable lighting for 2D rendering
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Only render if we have a reward to show
        if self.reward_text_timeout > 0:
            # Create reward indicator surface
            reward_surface = pygame.Surface((150, 40), pygame.SRCALPHA)
            
            # Background with fade based on timeout
            alpha = min(255, self.reward_text_timeout * 5)
            bg_color = (*self.reward_color[:3], alpha)
            
            pygame.draw.rect(
                reward_surface,
                bg_color,
                pygame.Rect(0, 0, 150, 40),
                border_radius=8
            )
            
            # Reward text
            reward_text = self.large_font.render(self.reward_text, True, (255, 255, 255))
            reward_surface.blit(reward_text, (75 - reward_text.get_width()//2, 10))
            
            # Position in center-top of screen
            reward_pos = (self.width//2 - 75, 50)
            
            # Convert Surface to OpenGL texture and render it
            reward_data = pygame.image.tostring(reward_surface, "RGBA", True)
            glRasterPos2d(reward_pos[0], reward_pos[1])
            glDrawPixels(150, 40, GL_RGBA, GL_UNSIGNED_BYTE, reward_data)
        
        # Render reward history graph
        if len(self.reward_history) > 1:
            graph_width = 200
            graph_height = 50
            graph_x = self.width - graph_width - 10
            graph_y = self.height - graph_height - 50
            
            # Create graph surface
            graph_surface = pygame.Surface((graph_width, graph_height), pygame.SRCALPHA)
            graph_surface.fill((0, 0, 0, 150))  # Semi-transparent black background
            
            # Find min and max rewards for scaling
            min_reward = min(min(self.reward_history), -0.1)  # At least -0.1
            max_reward = max(max(self.reward_history), 0.1)   # At least 0.1
            reward_range = max_reward - min_reward
            
            # Draw zero line
            zero_y = int(graph_height * (max_reward / reward_range))
            pygame.draw.line(
                graph_surface,
                (150, 150, 150),
                (0, zero_y),
                (graph_width, zero_y),
                1
            )
            
            # Draw reward history
            points = []
            for i, reward in enumerate(self.reward_history):
                x = int((i / len(self.reward_history)) * graph_width)
                y = int(graph_height * (max_reward - reward) / reward_range)
                points.append((x, y))
            
            # Connect points with lines
            if len(points) > 1:
                pygame.draw.lines(
                    graph_surface,
                    (0, 255, 0),  # Green line
                    False,
                    points,
                    2
                )
            
            # Draw graph title
            title = self.small_font.render("Reward History", True, (255, 255, 255))
            graph_surface.blit(title, (graph_width//2 - title.get_width()//2, 2))
            
            # Convert Surface to OpenGL texture and render it
            graph_data = pygame.image.tostring(graph_surface, "RGBA", True)
            glRasterPos2d(graph_x, graph_y)
            glDrawPixels(graph_width, graph_height, GL_RGBA, GL_UNSIGNED_BYTE, graph_data)
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def _set_camera_view(self):
        """Set the camera position and orientation based on current camera mode"""
        current_mode = self.camera_modes[self.current_camera_mode]
        
        if current_mode == "follow":
            # Follow camera - behind the agent
            cam_offset_x = -self.camera_distance * math.sin(math.radians(self.agent_rotation))
            cam_offset_y = -self.camera_distance * math.cos(math.radians(self.agent_rotation))
            
            gluLookAt(
                self.agent_pos[0] + cam_offset_x, 
                self.agent_pos[1] + cam_offset_y, 
                self.camera_height,  # Camera height
                self.agent_pos[0], 
                self.agent_pos[1], 
                0.7,  # Look at point (focused on agent)
                0, 0, 1  # Up vector
            )
            
        elif current_mode == "overview":
            # Top-down overview camera
            gluLookAt(
                5.0, 5.0, 15.0,  # Position high above center
                5.0, 5.0, 0.0,   # Look at center of environment
                0.0, 1.0, 0.0    # Up vector (now Y is up)
            )
            
        elif current_mode == "first_person":
            # First-person perspective
            # Calculate look-at point based on agent's rotation
            look_x = self.agent_pos[0] + 5.0 * math.sin(math.radians(self.agent_rotation))
            look_y = self.agent_pos[1] + 5.0 * math.cos(math.radians(self.agent_rotation))
            
            gluLookAt(
                self.agent_pos[0], 
                self.agent_pos[1], 
                1.5,  # Just above agent's head
                look_x, look_y, 1.0,  # Look ahead
                0, 0, 1  # Up vector
            )
            
        elif current_mode == "side_view":
            # Side view camera
            # Calculate perpendicular vector to agent's direction
            perp_x = 8.0 * math.sin(math.radians(self.agent_rotation + 90))
            perp_y = 8.0 * math.cos(math.radians(self.agent_rotation + 90))
            
            gluLookAt(
                self.agent_pos[0] + perp_x, 
                self.agent_pos[1] + perp_y, 
                4.0,  # Camera height
                self.agent_pos[0], 
                self.agent_pos[1], 
                0.5,  # Look at agent
                0, 0, 1  # Up vector
            )
    
    def _render_camera_mode_indicator(self):
        """Render an indicator showing the current camera mode"""
        # Switch to 2D projection
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, self.height, 0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable lighting for 2D rendering
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Create indicator surface
        indicator = pygame.Surface((150, 30), pygame.SRCALPHA)
        
        # Background
        pygame.draw.rect(
            indicator,
            (0, 0, 0, 150),  # Semi-transparent black
            pygame.Rect(0, 0, 150, 30),
            border_radius=5
        )
        
        # Camera mode text
        mode_text = self.small_font.render(
            f"Camera: {self.camera_modes[self.current_camera_mode].replace('_', ' ').title()}", 
            True, (255, 255, 255)
        )
        indicator.blit(mode_text, (10, 6))
        
        # Position in top-right
        indicator_pos = (self.width - 160, 170)
        
        # Convert Surface to OpenGL texture and render it
        indicator_data = pygame.image.tostring(indicator, "RGBA", True)
        glRasterPos2d(indicator_pos[0], indicator_pos[1])
        glDrawPixels(150, 30, GL_RGBA, GL_UNSIGNED_BYTE, indicator_data)
        
        # Draw key hint for camera mode switching
        hint_text = self.small_font.render("Press C to change camera", True, (255, 255, 255))
        hint_data = pygame.image.tostring(hint_text, "RGBA", True)
        glRasterPos2d(indicator_pos[0], indicator_pos[1] + 25)
        glDrawPixels(hint_text.get_width(), hint_text.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, hint_data)
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def _render_ui_overlay(self, collision_type, location_reached):
        """Render UI elements over the 3D scene"""
        # Switch to 2D orthographic projection
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, self.height, 0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable lighting and depth test for UI
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Create a surface for UI elements
        ui_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw stats panel background
        pygame.draw.rect(
            ui_surface,
            (240, 240, 240, 200),  # Semi-transparent light gray
            pygame.Rect(5, 5, 200, 120),  # Position and size
            border_radius=5
        )
        pygame.draw.rect(
            ui_surface,
            (100, 100, 100),  # Gray border
            pygame.Rect(5, 5, 200, 120),
            width=2,
            border_radius=5
        )
        
        # Render text information
        title_text = self.large_font.render("Hospital Navigator 3D", True, (0, 0, 0))
        steps_text = self.font.render(f"Steps: {self.step_count}/{self.max_steps}", True, (0, 0, 0))
        
        # Show reward with appropriate color
        if self.reward > 0:
            reward_color = (0, 128, 0)  # Green for positive
        elif self.reward < 0:
            reward_color = (200, 0, 0)  # Red for negative
        else:
            reward_color = (0, 0, 0)    # Black for zero
            
        reward_text = self.font.render(f"Reward: {self.reward:.2f}", True, reward_color)
        
        # Show cumulative reward
        if self.cumulative_reward > 0:
            cum_color = (0, 128, 0)  # Green for positive
        elif self.cumulative_reward < 0:
            cum_color = (200, 0, 0)  # Red for negative
        else:
            cum_color = (0, 0, 0)    # Black for zero
            
        cumulative_text = self.font.render(f"Total Score: {self.cumulative_reward:.2f}", True, cum_color)
        distance_text = self.font.render(f"Distance: {self._calculate_distance_to_target():.1f}", True, (0, 0, 0))
        
        # Mode text with appropriate color
        if self.mode == 'delivery':
            mode_color = (0, 100, 0)  # Dark green
        else:  # emergency
            mode_color = (150, 0, 0)  # Dark red
        mode_text = self.font.render(f"Mode: {self.mode.capitalize()}", True, mode_color)
        
        # Render all text elements to the surface
        ui_surface.blit(title_text, (15, 10))
        ui_surface.blit(steps_text, (15, 40))
        ui_surface.blit(reward_text, (15, 60))
        ui_surface.blit(cumulative_text, (15, 80))
        ui_surface.blit(mode_text, (15, 100))
        
        # Render status messages for collisions or location reached
        status_messages = []
        
        if location_reached:
            loc_label = self.room_labels.get(location_reached, location_reached.replace('_', ' ').title())
            status_text = f"In {loc_label}"
            
            if location_reached == 'delivery_room' and self.mode == 'delivery':
                status_messages.append((status_text + " (+50)", (0, 255, 0)))
            elif location_reached == 'emergency_room' and self.mode == 'emergency':
                status_messages.append((status_text + " (+50)", (0, 255, 0)))
            elif location_reached == 'prenatal_care':
                status_messages.append((status_text + " (+10)", (0, 200, 0)))
            elif location_reached == 'reception':
                status_messages.append((status_text + " (+5)", (0, 150, 0)))
            else:
                status_messages.append((status_text, (0, 0, 0)))
        
        if collision_type == 'staff':
            status_messages.append(("Collision with Medical Staff (-10)", (255, 0, 0)))
        elif collision_type == 'crowded':
            status_messages.append(("In Crowded Area (-5)", (255, 165, 0)))
        
        # Render status messages at the bottom of the screen
        for i, (text, color) in enumerate(status_messages):
            status_text = self.font.render(text, True, color)
            ui_surface.blit(status_text, (15, self.height - 30 - i*25))
        
        # Render user info and date at the bottom
        user_info_text = self.small_font.render(f"User: {self.current_user}", True, (50, 50, 50))
        date_text = self.small_font.render(f"Date: {self.current_date}", True, (50, 50, 50))
        ui_surface.blit(user_info_text, (5, self.height - 40))
        ui_surface.blit(date_text, (5, self.height - 20))
        
        # Draw controls help panel
        control_panel_width = 200
        control_panel_height = 140
        control_panel_x = 10
        control_panel_y = self.height - 190
        
        pygame.draw.rect(
            ui_surface,
            (0, 0, 0, 180),  # Semi-transparent black
            pygame.Rect(control_panel_x, control_panel_y, control_panel_width, control_panel_height),
            border_radius=5
        )
        
        # Controls title
        control_title = self.small_font.render("Controls", True, (255, 255, 255))
        ui_surface.blit(control_title, (control_panel_x + 10, control_panel_y + 10))
        
        # Control instructions
        controls = [
            "C - Change camera view",
            "M - Toggle auto movement",
            "Arrow keys - Manual control",
            "Q - Quit simulation"
        ]
        
        for i, control in enumerate(controls):
            control_text = self.small_font.render(control, True, (255, 255, 255))
            ui_surface.blit(control_text, (control_panel_x + 15, control_panel_y + 35 + i*25))
        
        # Convert Surface to OpenGL texture and render it
        ui_texture_data = pygame.image.tostring(ui_surface, "RGBA", True)
        glDrawPixels(self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, ui_texture_data)
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def _render_minimap(self):
        """Render a minimap showing room layout and positions"""
        # Switch to 2D projection
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, self.height, 0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable lighting for 2D rendering
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Create minimap surface
        mini_map_size = 150
        mini_map = pygame.Surface((mini_map_size, mini_map_size), pygame.SRCALPHA)
        mini_map.fill((220, 220, 220, 200))
        
        # Draw room areas on minimap
        for room_name, room_data in self.locations.items():
            # Skip the hallway_center as it's not a real room
            if room_name == 'hallway_center':
                continue
                
            pos = room_data['pos']
            size = room_data['size']
            
            # Convert world coordinates to minimap coordinates
            mini_x = int(pos[0] * mini_map_size / 10)
            mini_y = int((10 - pos[1]) * mini_map_size / 10)  # Flip Y axis
            mini_width = int(size[0] * mini_map_size / 10)
            mini_height = int(size[1] * mini_map_size / 10)
            
            # Choose color based on room type with enhanced colors
            if room_name == 'delivery_room':
                color = self.colors['delivery']
            elif room_name == 'emergency_room':
                color = self.colors['emergency']
            elif room_name == 'reception':
                color = self.colors['reception']
            elif room_name == 'prenatal_care':
                color = self.colors['hallway']
            else:
                color = self.colors['floor']
            
            # Draw room rectangle
            pygame.draw.rect(
                mini_map,
                color,
                pygame.Rect(
                    mini_x - mini_width//2,
                    mini_y - mini_height//2,
                    mini_width,
                    mini_height
                )
            )
            
            # Add room label
            if room_name in self.room_labels:
                label = self.small_font.render(
                    self.room_labels[room_name].split()[0],  # Just use first word to save space
                    True, 
                    (0, 0, 0)
                )
                mini_map.blit(
                    label, 
                    (mini_x - label.get_width()//2, mini_y - label.get_height()//2)
                )
            
            # Highlight target room with a border
            if (self.mode == 'delivery' and room_name == 'delivery_room') or \
               (self.mode == 'emergency' and room_name == 'emergency_room'):
                pygame.draw.rect(
                    mini_map,
                    (255, 255, 0),  # Yellow border
                    pygame.Rect(
                        mini_x - mini_width//2,
                        mini_y - mini_height//2,
                        mini_width,
                        mini_height
                    ),
                    width=2
                )
        
        # Draw automatic path
        if self.auto_movement and len(self.auto_path) > 0:
            for i in range(len(self.auto_path) - 1):
                start_x = int(self.auto_path[i][0] * mini_map_size / 10)
                start_y = int((10 - self.auto_path[i][1]) * mini_map_size / 10)
                end_x = int(self.auto_path[i+1][0] * mini_map_size / 10)
                end_y = int((10 - self.auto_path[i+1][1]) * mini_map_size / 10)
                
                # Draw path segments
                pygame.draw.line(
                    mini_map,
                    (50, 50, 50, 120),  # Semi-transparent dark gray
                    (start_x, start_y),
                    (end_x, end_y),
                    1
                )
            
            # Highlight current path segment
            if self.current_path_index < len(self.auto_path) - 1:
                start_x = int(self.auto_path[self.current_path_index][0] * mini_map_size / 10)
                start_y = int((10 - self.auto_path[self.current_path_index][1]) * mini_map_size / 10)
                end_x = int(self.auto_path[self.current_path_index+1][0] * mini_map_size / 10)
                end_y = int((10 - self.auto_path[self.current_path_index+1][1]) * mini_map_size / 10)
                
                pygame.draw.line(
                    mini_map,
                    (255, 255, 0, 180),  # Yellow line
                    (start_x, start_y),
                    (end_x, end_y),
                    2
                )
        
        # Draw staff on minimap
        for staff in self.medical_staff:
            # Convert coordinates
            mini_x = int(staff['pos'][0] * mini_map_size / 10)
            mini_y = int((10 - staff['pos'][1]) * mini_map_size / 10)
            
            # Draw staff dot
            color = (255, 0, 0) if staff['type'] == 'doctor' else (0, 100, 255)
            pygame.draw.circle(mini_map, color, (mini_x, mini_y), 3)
        
        # Draw agent on minimap
        mini_agent_x = int(self.agent_pos[0] * mini_map_size / 10)
        mini_agent_y = int((10 - self.agent_pos[1]) * mini_map_size / 10)
        pygame.draw.circle(mini_map, self.colors['agent'], (mini_agent_x, mini_agent_y), 5)
        
        # Draw direction indicator
        direction_x = mini_agent_x + int(8 * math.sin(math.radians(self.agent_rotation)))
        direction_y = mini_agent_y - int(8 * math.cos(math.radians(self.agent_rotation)))
        pygame.draw.line(mini_map, (0, 0, 0), (mini_agent_x, mini_agent_y), (direction_x, direction_y), 2)
        
        # Draw target on minimap
        mini_target_x = int(self.target_pos[0] * mini_map_size / 10)
        mini_target_y = int((10 - self.target_pos[1]) * mini_map_size / 10)
        pygame.draw.circle(mini_map, self.colors['target'], (mini_target_x, mini_target_y), 4)
        
        # Draw minimap border
        pygame.draw.rect(mini_map, (0, 0, 0), pygame.Rect(0, 0, mini_map_size, mini_map_size), width=2)
        
        # Draw title
        title = self.small_font.render("Map", True, (0, 0, 0))
        mini_map.blit(title, (mini_map_size//2 - title.get_width()//2, 5))
        
        # Position minimap in top-right corner
        mini_map_pos = (self.width - mini_map_size - 10, 10)
        
        # Convert Surface to OpenGL texture and render it
        mini_map_data = pygame.image.tostring(mini_map, "RGBA", True)
        glRasterPos2d(mini_map_pos[0], mini_map_pos[1])
        glDrawPixels(mini_map_size, mini_map_size, GL_RGBA, GL_UNSIGNED_BYTE, mini_map_data)
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def _render_color_key(self):
        """Render a color key showing what different colors represent"""
        # Switch to 2D projection
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, self.height, 0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable lighting for 2D rendering
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Create color key panel
        key_panel_width = 120
        key_panel_height = 220  # Increased height for more items
        key_panel_x = 10
        key_panel_y = 130
        
        color_key = pygame.Surface((key_panel_width, key_panel_height), pygame.SRCALPHA)
        
        # Panel background
        pygame.draw.rect(
            color_key,
            (240, 240, 240, 200),  # Semi-transparent light gray
            pygame.Rect(0, 0, key_panel_width, key_panel_height),
            border_radius=5
        )
        pygame.draw.rect(
            color_key,
            (100, 100, 100),  # Gray border
            pygame.Rect(0, 0, key_panel_width, key_panel_height),
            width=2,
            border_radius=5
        )
        
        # Draw title
        key_title = self.small_font.render("Color Key", True, (0, 0, 0))
        color_key.blit(key_title, (key_panel_width//2 - key_title.get_width()//2, 5))
        
        # List of items in the key (expanded)
        key_items = [
            ('Patient', 'agent'),
            ('Doctor', (255, 0, 0)),  
            ('Nurse', (0, 100, 255)),  
            ('Emergency', 'emergency'),
            ('Delivery', 'delivery'),
            ('Reception', 'reception'),
            ('Prenatal', 'hallway'),
            ('Bed', 'bed'),
            ('Target', 'target'),
            ('Pos. Reward', 'positive_reward'),
            ('Neg. Reward', 'negative_reward'),
            ('Crowded', (255, 200, 100))
        ]
        
        # Draw each item in the key
        y_offset = 25
        for i, (label, color_key_value) in enumerate(key_items):
            # Color box
            if isinstance(color_key_value, str):
                # Get color from predefined colors
                box_color = self.colors[color_key_value]
            else:
                # Use direct RGB value
                box_color = color_key_value
                
            pygame.draw.rect(
                color_key,
                box_color,
                pygame.Rect(10, y_offset + i*16, 12, 12),
                border_radius=2
            )
            
            # Label
            label_text = self.small_font.render(label, True, (0, 0, 0))
            color_key.blit(label_text, (30, y_offset + i*16 - 2))
        
        # Convert Surface to OpenGL texture and render it
        key_data = pygame.image.tostring(color_key, "RGBA", True)
        glRasterPos2d(key_panel_x, key_panel_y)
        glDrawPixels(key_panel_width, key_panel_height, GL_RGBA, GL_UNSIGNED_BYTE, key_data)
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def _render_directional_guidance(self):
        """Render a directional indicator pointing to the target"""
        # Switch to 2D projection
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, self.height, 0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable lighting for 2D rendering
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Create guidance surface
        guidance_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Calculate direction vector from agent to target
        dx = self.target_pos[0] - self.agent_pos[0]
        dy = self.target_pos[1] - self.agent_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # Convert 3D world coordinates to 2D screen coordinates (approximation)
        screen_width, screen_height = self.width, self.height
        agent_screen_x = screen_width // 2
        agent_screen_y = screen_height // 2
        
        # Normalize direction and scale
        scale_factor = 80
        if distance > 0:
            dx = dx / distance * scale_factor
            dy = dy / distance * scale_factor
        
        # Arrow end point - account for the fact that the Y-axis is flipped in screen coordinates
        arrow_x = agent_screen_x + dx
        arrow_y = agent_screen_y - dy  # Flip Y
        
        # Choose color based on mode
        if self.mode == 'delivery':
            arrow_color = (0, 200, 0, 180)  # Semi-transparent green
        else:
            arrow_color = (200, 0, 0, 180)  # Semi-transparent red
        
        # Only show directional guidance in first-person mode
        if self.camera_modes[self.current_camera_mode] == "first_person":
            # Draw arrow
            arrow_width = 5
            pygame.draw.line(guidance_surface, arrow_color, 
                            (agent_screen_x, agent_screen_y), 
                            (arrow_x, arrow_y), arrow_width)
            
            # Draw arrowhead
            arrow_size = 12
            angle = math.atan2(-dy, dx)  # Y is flipped
            
            # Calculate arrowhead points
            p1_x = arrow_x - arrow_size * math.cos(angle - math.pi/6)
            p1_y = arrow_y - arrow_size * math.sin(angle - math.pi/6)
            p2_x = arrow_x - arrow_size * math.cos(angle + math.pi/6)
            p2_y = arrow_y - arrow_size * math.sin(angle + math.pi/6)
            
            # Draw arrowhead
            pygame.draw.polygon(guidance_surface, arrow_color, 
                            [(arrow_x, arrow_y), (p1_x, p1_y), (p2_x, p2_y)])
            
            # Draw distance text
            distance_text = self.small_font.render(f"{distance:.1f}m", True, (0, 0, 0))
            
            # Position text along the arrow
            text_x = agent_screen_x + dx * 0.6 - distance_text.get_width() // 2
            text_y = agent_screen_y - dy * 0.6 - distance_text.get_height() // 2
            
            # Draw background for text
            bg_rect = pygame.Rect(text_x - 5, text_y - 2, 
                                distance_text.get_width() + 10, 
                                distance_text.get_height() + 4)
            pygame.draw.rect(guidance_surface, (255, 255, 255, 180), bg_rect)
            
            # Draw text
            guidance_surface.blit(distance_text, (text_x, text_y))
            
            # Draw target room label
            target_room = 'delivery_room' if self.mode == 'delivery' else 'emergency_room'
            target_label = f"To {self.room_labels[target_room]}"
            target_text = self.small_font.render(target_label, True, (0, 0, 0))
            
            # Position target label
            label_x = arrow_x - target_text.get_width() // 2
            label_y = arrow_y + 5
            
            # Draw background for label
            label_bg_rect = pygame.Rect(label_x - 5, label_y - 2, 
                                    target_text.get_width() + 10, 
                                    target_text.get_height() + 4)
            pygame.draw.rect(guidance_surface, (255, 255, 255, 180), label_bg_rect)
            
            # Draw label
            guidance_surface.blit(target_text, (label_x, label_y))
        
        # Convert Surface to OpenGL texture and render it
        guidance_data = pygame.image.tostring(guidance_surface, "RGBA", True)
        glDrawPixels(self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, guidance_data)
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def draw_cube(self, x, y, z, width, depth, height):
        """Draw a cube with specified dimensions and position"""
        w2, d2, h2 = width/2, depth/2, height/2
        
        glPushMatrix()
        glTranslatef(x, y, z + h2)
        
        glBegin(GL_QUADS)
        # Front face
        glNormal3f(0, 1, 0)
        glVertex3f(-w2, d2, -h2)
        glVertex3f(w2, d2, -h2)
        glVertex3f(w2, d2, h2)
        glVertex3f(-w2, d2, h2)
        
        # Back face
        glNormal3f(0, -1, 0)
        glVertex3f(-w2, -d2, -h2)
        glVertex3f(w2, -d2, -h2)
        glVertex3f(w2, -d2, h2)
        glVertex3f(-w2, -d2, h2)
        
        # Right face
        glNormal3f(1, 0, 0)
        glVertex3f(w2, -d2, -h2)
        glVertex3f(w2, d2, -h2)
        glVertex3f(w2, d2, h2)
        glVertex3f(w2, -d2, h2)
        
        # Left face
        glNormal3f(-1, 0, 0)
        glVertex3f(-w2, -d2, -h2)
        glVertex3f(-w2, d2, -h2)
        glVertex3f(-w2, d2, h2)
        glVertex3f(-w2, -d2, h2)
        
        # Top face
        glNormal3f(0, 0, 1)
        glVertex3f(-w2, -d2, h2)
        glVertex3f(w2, -d2, h2)
        glVertex3f(w2, d2, h2)
        glVertex3f(-w2, d2, h2)
        
        # Bottom face
        glNormal3f(0, 0, -1)
        glVertex3f(-w2, -d2, -h2)
        glVertex3f(w2, -d2, -h2)
        glVertex3f(w2, d2, -h2)
        glVertex3f(-w2, d2, -h2)
        glEnd()
        
        glPopMatrix()
    
    def draw_sphere(self, x, y, z, radius, slices=16, stacks=16):
        """Draw a sphere at the specified position"""
        glPushMatrix()
        glTranslatef(x, y, z)
        quadric = gluNewQuadric()
        gluSphere(quadric, radius, slices, stacks)
        gluDeleteQuadric(quadric)
        glPopMatrix()
    
    def draw_bed(self, x, y, z):
        """Draw a hospital bed at the specified position"""
        # Bed frame
        glColor3fv(self.colors_3d['bed_frame'])
        self.draw_cube(x, y, z, 1.5, 0.8, 0.3)
        
        # Mattress
        glColor3fv(self.colors_3d['bed'])
        self.draw_cube(x, y, z + 0.3, 1.4, 0.7, 0.1)
        
        # Pillow
        glColor3f(1, 1, 1)
        self.draw_cube(x + 0.5, y, z + 0.4, 0.3, 0.6, 0.08)
        
        # Add bedside table
        glColor3f(0.7, 0.6, 0.5)
        self.draw_cube(x - 0.8, y, z, 0.4, 0.4, 0.5)
    
    def create_hospital_display_list(self):
        """Create a display list for the hospital environment"""
        display_list = glGenLists(1)
        glNewList(display_list, GL_COMPILE)
        
        # Draw floor with grid
        glColor3fv(self.colors_3d['floor'])
        glBegin(GL_QUADS)
        glNormal3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(10, 0, 0)
        glVertex3f(10, 10, 0)
        glVertex3f(0, 10, 0)
        glEnd()
        
        # Draw grid lines
        glColor3f(0.7, 0.7, 0.7)
        glBegin(GL_LINES)
        for i in range(11):
            glVertex3f(i, 0, 0.01)
            glVertex3f(i, 10, 0.01)
            glVertex3f(0, i, 0.01)
            glVertex3f(10, i, 0.01)
        glEnd()
        
        # Draw walls
        glColor3fv(self.colors_3d['wall'])
        
        # North wall
        glBegin(GL_QUADS)
        glNormal3f(0, -1, 0)
        glVertex3f(0, 10, 0)
        glVertex3f(10, 10, 0)
        glVertex3f(10, 10, 3)
        glVertex3f(0, 10, 3)
        glEnd()
        
        # East wall
        glBegin(GL_QUADS)
        glNormal3f(-1, 0, 0)
        glVertex3f(10, 0, 0)
        glVertex3f(10, 10, 0)
        glVertex3f(10, 10, 3)
        glVertex3f(10, 0, 3)
        glEnd()
        
        # South wall
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(10, 0, 0)
        glVertex3f(10, 0, 3)
        glVertex3f(0, 0, 3)
        glEnd()
        
        # West wall
        glBegin(GL_QUADS)
        glNormal3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 10, 0)
        glVertex3f(0, 10, 3)
        glVertex3f(0, 0, 3)
        glEnd()
        
        # Draw special areas with transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Delivery room
        glColor4fv(self.colors_3d['delivery_room'])
        self.draw_cube(7, 7, 0, 3, 3, 0.1)
        
        # Prenatal care
        glColor4fv(self.colors_3d['prenatal_care'])
        self.draw_cube(3, 8, 0, 2, 2, 0.1)
        
        # Emergency room
        glColor4fv(self.colors_3d['emergency_room'])
        self.draw_cube(8, 2, 0, 2, 2, 0.1)
        
        # Crowded areas
        glColor4fv(self.colors_3d['crowded_area'])
        for area in self.crowded_areas:
            self.draw_cube(area['pos'][0], area['pos'][1], 0, 
                          area['radius']*2, area['radius']*2, 0.1)
        
        glDisable(GL_BLEND)
        
        # Draw beds in various locations
        for bed_pos in self.beds:
            self.draw_bed(*bed_pos)
        
        # Draw reception desk
        glColor3fv(self.colors_3d['reception'])
        self.draw_cube(2, 8, 0, 2, 1, 1)
        
        # Add computer at reception
        glColor3f(0.2, 0.2, 0.2)
        self.draw_cube(2, 8.3, 1.1, 0.4, 0.3, 0.3)
        
        # Add chairs in waiting area
        glColor3f(0.4, 0.4, 0.8)
        for i in range(3):
            self.draw_cube(4 + i*0.8, 5, 0, 0.5, 0.5, 0.5)
        
        # Add medical equipment
        glColor3fv(self.colors_3d['medical_equipment'])
        self.draw_cube(7, 2, 0, 0.5, 0.5, 0.8)  # Ultrasound machine
        self.draw_cube(3, 7, 0, 0.8, 0.5, 0.5)  # Medicine cabinet
        
        # Add room labels in 3D
        self._draw_room_labels()
        
        glEndList()
        return display_list
    
    def _draw_room_labels(self):
        """Draw 3D text labels for rooms"""
        # This is a simplified approach as true 3D text is complex in raw OpenGL
        # Usually would use specialized libraries for 3D text
        
        # For delivery room
        glColor3f(0, 0.7, 0)
        self.draw_cube(7, 7, 0.15, 1.5, 0.3, 0.1)
        
        # For emergency room
        glColor3f(0.7, 0, 0)
        self.draw_cube(8, 2, 0.15, 1.5, 0.3, 0.1)
        
        # For reception
        glColor3f(0.5, 0.3, 0.1)
        self.draw_cube(2, 8, 1.1, 1.0, 0.2, 0.1)
        
        # For prenatal care
        glColor3f(0.2, 0.2, 0.7)
        self.draw_cube(3, 8, 0.15, 1.0, 0.2, 0.1)
    
    def create_staff_display_list(self):
        """Create display lists for medical staff"""
        doctor_list = glGenLists(1)
        glNewList(doctor_list, GL_COMPILE)
        # Doctor body (white coat)
        glColor3f(1, 1, 1)
        self.draw_cube(0, 0, 0.5, 0.4, 0.3, 0.6)
        
        # Head
        glColor3fv(self.colors_3d['skin'])
        self.draw_sphere(0, 0, 1.2, 0.15)
        
        # Simple stethoscope representation
        glColor3f(0.2, 0.2, 0.2)
        glBegin(GL_LINES)
        glVertex3f(0.1, 0, 1.1)
        glVertex3f(-0.1, 0, 1.1)
        glEnd()
        glEndList()
        
        nurse_list = glGenLists(1)
        glNewList(nurse_list, GL_COMPILE)
        # Nurse body (blue uniform)
        glColor3f(0.2, 0.4, 0.8)
        self.draw_cube(0, 0, 0.5, 0.4, 0.3, 0.6)
        
        # Head
        glColor3fv(self.colors_3d['skin'])
        self.draw_sphere(0, 0, 1.2, 0.15)
        
        # Nurse cap
        glColor3f(1, 1, 1)
        self.draw_cube(0, 0, 1.3, 0.2, 0.3, 0.05)
        glEndList()
        
        return {'doctor': doctor_list, 'nurse': nurse_list}
    
    def create_agent_display_list(self):
        """Create a display list for the pregnant agent"""
        display_list = glGenLists(1)
        glNewList(display_list, GL_COMPILE)
        
        # Draw the body (torso)
        glColor3fv(self.colors_3d['dress'])
        self.draw_cube(0, 0, 0.5, 0.4, 0.3, 0.6)
        
        # Draw the belly (pregnant)
        glColor3fv(self.colors_3d['belly'])
        self.draw_sphere(0, 0.15, 0.7, 0.3)
        
        # Draw the head
        glColor3fv(self.colors_3d['skin'])
        self.draw_sphere(0, 0, 1.2, 0.15)
        
        # Add hair
        glColor3f(0.2, 0.1, 0.05)
        self.draw_sphere(0, 0, 1.28, 0.12)
        
        # Draw arms
        glPushMatrix()
        glTranslatef(0.25, 0, 0.9)
        
        # Right arm
        glColor3fv(self.colors_3d['skin'])
        self.draw_cube(0, 0, 0, 0.1, 0.1, 0.3)
        
        # Left arm
        glTranslatef(-0.5, 0, 0)
        self.draw_cube(0, 0, 0, 0.1, 0.1, 0.3)
        glPopMatrix()
        
        # Draw legs
        glPushMatrix()
        glTranslatef(0.12, 0, 0.2)
        
        # Right leg
        glColor3fv(self.colors_3d['skin'])
        self.draw_cube(0, 0, 0, 0.1, 0.1, 0.4)
        
        # Left leg
        glTranslatef(-0.24, 0, 0)
        self.draw_cube(0, 0, 0, 0.1, 0.1, 0.4)
        glPopMatrix()
        
        glEndList()
        return display_list
    
    def close(self):
        if self.render_mode is not None:
            pygame.quit()
    
    def change_camera_mode(self):
        """Manually change camera mode"""
        self.current_camera_mode = (self.current_camera_mode + 1) % len(self.camera_modes)
        # Reset the camera switch timer when manually changing
        self.camera_switch_timer = 0

def main():
    try:
        # Create and run simulation with updated user information
        env = HospitalSimulation3D(render_mode='human', mode='delivery',
                                  auto_movement=True)
        env.current_date = "2025-04-03 15:09:46"  # Updated time from user
        env.current_user = "Emmanuel-Begati"
        
        obs, info = env.reset()
        
        done = False
        total_reward = 0
        steps = 0
        
        print("Initial observation shape:", obs.shape)
        print("Action space:", env.action_space)
        print("Observation space:", env.observation_space)
        
        print("\nEnvironment test running. Controls:")
        print("- Press C to change camera view")
        print("- Press M to toggle auto-movement")
        print("- Press arrow keys to control agent when auto-movement is off")
        print("- Press Q to quit")
        print("Target: Reach the", "Delivery Room" if env.mode == 'delivery' else "Emergency Room")
        
        # Control loop - runs indefinitely until manually quit
        auto_movement_enabled = True
        while not done:
            action = 0  # Default action
            
            # Process events for human control and camera switching
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        # Change camera view
                        env.change_camera_mode()
                        print(f"Camera mode changed to: {env.camera_modes[env.current_camera_mode]}")
                    elif event.key == pygame.K_m:
                        # Toggle auto movement
                        auto_movement_enabled = not auto_movement_enabled
                        env.auto_movement = auto_movement_enabled
                        print(f"Auto-movement: {'Enabled' if auto_movement_enabled else 'Disabled'}")
                    elif event.key == pygame.K_UP and not auto_movement_enabled:
                        action = 0  # Up
                    elif event.key == pygame.K_RIGHT and not auto_movement_enabled:
                        action = 1  # Right
                    elif event.key == pygame.K_DOWN and not auto_movement_enabled:
                        action = 2  # Down
                    elif event.key == pygame.K_LEFT and not auto_movement_enabled:
                        action = 3  # Left
                    elif event.key == pygame.K_q:
                        done = True
            
            # Take action - environment will run indefinitely now
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update stats
            steps += 1
            total_reward += reward
            
            # Provide feedback every 100 steps
            if steps % 100 == 0:
                print(f"Step {steps}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
                if info.get('location'):
                    print(f"Current location: {info['location']}")
            
            # Render environment
            env.render()
            
        # Display final results
        print(f"\nVisualization completed after {steps} steps")
        print(f"Total reward: {total_reward:.2f}")
        
        env.close()
        print("Environment visualization successful!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        pygame.quit()
        print("Exited gracefully")

if __name__ == "__main__":
    main()  