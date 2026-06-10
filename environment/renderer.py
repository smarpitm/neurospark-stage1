# pyrefly: ignore [missing-import]
import pygame
import sys
import numpy as np
import random
import math

class Renderer:
    # Catppuccin Mocha-inspired color palette
    COLOR_BG = (30, 30, 46)          # Base background
    COLOR_PANEL = (17, 17, 27)       # Side panel background
    COLOR_GRID = (49, 50, 68)        # Grid lines
    COLOR_GRID_INNER = (31, 32, 44)  # Inner grid background
    COLOR_AGENT = (137, 180, 250)    # Blue agent
    COLOR_AGENT_GLOW = (137, 180, 250, 50)  # Agent translucent glow
    COLOR_GOAL = (166, 227, 161)     # Green goal
    COLOR_GOAL_GLOW = (166, 227, 161, 60)   # Goal translucent glow
    COLOR_WALL = (243, 139, 168)     # Red for boundary/collision
    COLOR_FOV = (250, 179, 135)      # Orange for FOV patch
    COLOR_TRAIL = (116, 199, 236, 80) # Cyan trail line
    COLOR_TEXT = (205, 214, 244)     # Light gray text
    COLOR_TEXT_DIM = (147, 153, 178) # Muted text

    # Neural heatmap population colors
    COLOR_SENSORY = (166, 227, 161)      # Green  (#a6e3a1)
    COLOR_RECURRENT = (137, 180, 250)    # Blue   (#89b4fa)
    COLOR_MOTOR = (243, 139, 168)        # Red    (#f38ba8)
    COLOR_NEURON_DIM = (20, 20, 30)      # Dim background for silent neurons
    COLOR_OVERLAY_BG = (17, 17, 27, 180) # Semi-transparent overlay background

    def __init__(self, env, cell_size=50):
        self.env = env
        self.cell_size = cell_size
        self.grid_pixel_size = env.grid_size * cell_size
        
        # Dimensions
        self.grid_offset_x = 40
        self.grid_offset_y = 50
        self.panel_width = 400
        
        self.width = self.grid_pixel_size + self.grid_offset_x * 2 + self.panel_width
        self.height = max(self.grid_pixel_size + self.grid_offset_y * 2, 700)
        
        pygame.init()
        pygame.display.set_caption("NeuroSpark Stage 1: Active Inference Grid World")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Font initialization
        pygame.font.init()
        # Fallback list of clean sans fonts
        font_names = ["segoeui", "arial", "helvetica", "sans-serif"]
        self.font_title = self.get_font(font_names, 20, bold=True)
        self.font_subtitle = self.get_font(font_names, 16, bold=True)
        self.font_body = self.get_font(font_names, 14)
        self.font_mono = self.get_font(["consolas", "courier", "monospace"], 12)
        
        # Smooth movement state
        self.visual_x = float(env.agent_pos[0])
        self.visual_y = float(env.agent_pos[1])
        
        # Game stats
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.trail = [tuple(env.agent_pos)]
        self.collision_flash = 0 # frame count for red flash effect on collision

        # SNN neural data (updated each step via update_snn_data)
        self.snn_data = None  # dict or None

    def get_font(self, names, size, bold=False):
        for name in names:
            try:
                return pygame.font.SysFont(name, size, bold=bold)
            except:
                pass
        return pygame.font.Font(None, size)

    def update_snn_data(self, data):
        """
        Accept per-step SNN diagnostic data for visualization.

        :param data: dict with keys:
            - sensory_spikes:  np.ndarray (100,) – spike counts per sensory neuron
            - recurrent_spikes: np.ndarray (800,) – spike counts per recurrent neuron
            - motor_spikes:    np.ndarray (100,) – spike counts per motor neuron
            - motor_total:     int   – total motor spike count
            - action_name:     str   – current action (e.g. "RIGHT")
            - free_energy:     float – current free energy value
            - motor_firing_pct: float – percentage of motor neurons that fired
            - step:            int   – current step number
        """
        self.snn_data = data

    def draw_glowing_circle(self, surface, color_glow, center, radius, pulse_val=0.0):
        # Draw soft translucent outer rings
        glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        center_local = (radius * 2, radius * 2)
        
        num_layers = 5
        base_alpha = color_glow[3]
        for i in range(num_layers, 0, -1):
            r = radius + (i * 4) + int(pulse_val * 6)
            alpha = int(base_alpha * (1.0 - i / (num_layers + 1)))
            pygame.draw.circle(glow_surf, (color_glow[0], color_glow[1], color_glow[2], alpha), center_local, r)
            
        surface.blit(glow_surf, (center[0] - radius * 2, center[1] - radius * 2))

    def run(self):
        # Standalone game loop
        running = True
        last_random_move = pygame.time.get_ticks()
        random_delay_ms = 400 # speed of random agent movement

        while running:
            dt = self.clock.tick(60)
            current_time = pygame.time.get_ticks()
            action_taken = None

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_UP:
                        action_taken = self.env.UP
                    elif event.key == pygame.K_DOWN:
                        action_taken = self.env.DOWN
                    elif event.key == pygame.K_LEFT:
                        action_taken = self.env.LEFT
                    elif event.key == pygame.K_RIGHT:
                        action_taken = self.env.RIGHT

            # If no manual control, take random action after delay
            if action_taken is None and (current_time - last_random_move > random_delay_ms):
                action_taken = random.choice(self.env.get_actions())
                last_random_move = current_time

            # Step the environment
            if action_taken is not None:
                action_names = {self.env.UP: "UP", self.env.DOWN: "DOWN", self.env.LEFT: "LEFT", self.env.RIGHT: "RIGHT"}
                self.last_action = action_names.get(action_taken)
                
                obs, reward, done = self.env.step(action_taken)
                self.cumulative_reward += reward
                self.last_reward = reward
                self.step_count += 1
                
                # Check for collision
                if reward == -1.0:
                    self.collision_flash = 10  # 10 frames of red flash
                else:
                    curr_pos = tuple(self.env.agent_pos)
                    if not self.trail or self.trail[-1] != curr_pos:
                        self.trail.append(curr_pos)

                if done:
                    print(f"Goal Reached in {self.step_count} steps! Total Reward: {self.cumulative_reward:.2f}")
                    time_sleep = 1.0
                    self.draw(render_only=True)
                    pygame.display.flip()
                    pygame.time.wait(int(time_sleep * 1000))
                    # Reset
                    self.env.reset()
                    self.step_count = 0
                    self.cumulative_reward = 0.0
                    self.last_action = None
                    self.last_reward = 0.0
                    self.trail = [tuple(self.env.agent_pos)]
                    self.visual_x = float(self.env.agent_pos[0])
                    self.visual_y = float(self.env.agent_pos[1])
                    last_random_move = pygame.time.get_ticks()

            # Smooth agent movement animation
            target_x, target_y = self.env.agent_pos
            self.visual_x += (target_x - self.visual_x) * 0.22
            self.visual_y += (target_y - self.visual_y) * 0.22

            # Render
            self.draw()
            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def draw(self, render_only=False):
        # 1. Background
        self.screen.fill(self.COLOR_BG)
        
        # Draw Grid Background Area
        grid_rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y, self.grid_pixel_size, self.grid_pixel_size)
        pygame.draw.rect(self.screen, self.COLOR_GRID_INNER, grid_rect)
        
        # Pulse animation factor using sine wave
        pulse_val = 0.5 * (1.0 + math.sin(pygame.time.get_ticks() / 150.0))

        # 2. Draw Trajectory Trail
        if len(self.trail) > 1:
            points = []
            for pos in self.trail:
                px = self.grid_offset_x + pos[0] * self.cell_size + self.cell_size // 2
                py = self.grid_offset_y + pos[1] * self.cell_size + self.cell_size // 2
                points.append((px, py))
            
            # Draw trail line
            pygame.draw.lines(self.screen, self.COLOR_TRAIL, False, points, 3)
            # Draw nodes along trail
            for pt in points[:-1]:
                pygame.draw.circle(self.screen, (self.COLOR_TRAIL[0], self.COLOR_TRAIL[1], self.COLOR_TRAIL[2], 120), pt, 4)

        # 3. Draw Grid Lines
        for i in range(self.env.grid_size + 1):
            # Vertical
            start_v = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y)
            end_v = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y + self.grid_pixel_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_v, end_v, 1)
            # Horizontal
            start_h = (self.grid_offset_x, self.grid_offset_y + i * self.cell_size)
            end_h = (self.grid_offset_x + self.grid_pixel_size, self.grid_offset_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_h, end_h, 1)

        # 4. Draw Goal (9,9)
        goal_x, goal_y = self.env.goal_pos
        goal_center = (
            self.grid_offset_x + goal_x * self.cell_size + self.cell_size // 2,
            self.grid_offset_y + goal_y * self.cell_size + self.cell_size // 2
        )
        self.draw_glowing_circle(self.screen, self.COLOR_GOAL_GLOW, goal_center, 12, pulse_val)
        pygame.draw.circle(self.screen, self.COLOR_GOAL, goal_center, 10)
        pygame.draw.circle(self.screen, (255, 255, 255), goal_center, 4)

        # 5. Draw 3x3 FOV Box Highlight
        ax, ay = self.env.agent_pos
        fov_rect = pygame.Rect(
            self.grid_offset_x + (ax - 1) * self.cell_size,
            self.grid_offset_y + (ay - 1) * self.cell_size,
            self.cell_size * 3,
            self.cell_size * 3
        )
        # Clip to grid boundary visually
        fov_rect = fov_rect.clip(grid_rect)
        pygame.draw.rect(self.screen, self.COLOR_FOV, fov_rect, 2)
        
        # 6. Draw Agent (smooth visual position)
        agent_center = (
            int(self.grid_offset_x + self.visual_x * self.cell_size + self.cell_size // 2),
            int(self.grid_offset_y + self.visual_y * self.cell_size + self.cell_size // 2)
        )
        agent_radius = 12
        
        # Draw flash on collision
        if self.collision_flash > 0:
            if not render_only:
                self.collision_flash -= 1
            pygame.draw.rect(self.screen, self.COLOR_WALL, grid_rect, 3) # flash grid border
            self.draw_glowing_circle(self.screen, (self.COLOR_WALL[0], self.COLOR_WALL[1], self.COLOR_WALL[2], 120), agent_center, agent_radius + 4, 1.0)
            pygame.draw.circle(self.screen, self.COLOR_WALL, agent_center, agent_radius)
        else:
            self.draw_glowing_circle(self.screen, self.COLOR_AGENT_GLOW, agent_center, agent_radius, pulse_val)
            pygame.draw.circle(self.screen, self.COLOR_AGENT, agent_center, agent_radius)
        
        # Draw agent core/eye direction based on goal or just center
        pygame.draw.circle(self.screen, (255, 255, 255), agent_center, 4)

        # 6b. Draw on-grid SNN overlay (motor spikes + action)
        self._draw_grid_overlay()

        # 7. Draw Side Panel & HUD
        panel_rect = pygame.Rect(self.width - self.panel_width, 0, self.panel_width, self.height)
        pygame.draw.rect(self.screen, self.COLOR_PANEL, panel_rect)
        # Border separation
        pygame.draw.line(self.screen, self.COLOR_GRID, (self.width - self.panel_width, 0), (self.width - self.panel_width, self.height), 2)

        # Title
        title_y = 30
        title_surf = self.font_title.render("NEUROSPARK SNN", True, self.COLOR_AGENT)
        self.screen.blit(title_surf, (self.width - self.panel_width + 25, title_y))
        
        subtitle_surf = self.font_subtitle.render("Digital Twin: Grid World", True, self.COLOR_TEXT)
        self.screen.blit(subtitle_surf, (self.width - self.panel_width + 25, title_y + 28))

        # Stats
        stats_y = 100
        stats = [
            ("Step Count:", f"{self.step_count}"),
            ("Agent Pos:", f"({ax}, {ay})"),
            ("Goal Pos:", f"({goal_x}, {goal_y})"),
            ("Last Action:", f"{self.last_action}"),
            ("Last Reward:", f"{self.last_reward:+.2f}"),
            ("Total Reward:", f"{self.cumulative_reward:.2f}"),
        ]

        for label, val in stats:
            lbl_surf = self.font_body.render(label, True, self.COLOR_TEXT_DIM)
            val_surf = self.font_mono.render(val, True, self.COLOR_TEXT)
            self.screen.blit(lbl_surf, (self.width - self.panel_width + 25, stats_y))
            self.screen.blit(val_surf, (self.width - self.panel_width + 140, stats_y))
            stats_y += 24

        # 8. Render 3x3 FOV Local Patch Grid on HUD
        hud_fov_y = 265
        fov_title = self.font_subtitle.render("Agent FOV (3x3 Patch)", True, self.COLOR_FOV)
        self.screen.blit(fov_title, (self.width - self.panel_width + 25, hud_fov_y))
        
        obs = self.env.get_observation()
        # The first 9 values are the patch
        patch_vals = obs[:9]
        
        # Draw the 3x3 mini grid
        mini_cell_sz = 20
        mini_grid_offset_x = self.width - self.panel_width + 25
        mini_grid_offset_y = hud_fov_y + 25
        
        for r in range(3):
            for c in range(3):
                val = patch_vals[r * 3 + c]
                cell_rect = pygame.Rect(
                    mini_grid_offset_x + c * mini_cell_sz,
                    mini_grid_offset_y + r * mini_cell_sz,
                    mini_cell_sz,
                    mini_cell_sz
                )
                
                # Colors: Red if wall (1.0), deep blue/purple if agent itself (center r=1, c=1), else dark gray
                if val == 1.0:
                    c_col = self.COLOR_WALL
                elif r == 1 and c == 1:
                    c_col = self.COLOR_AGENT
                else:
                    c_col = self.COLOR_GRID_INNER
                    
                pygame.draw.rect(self.screen, c_col, cell_rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, cell_rect, 1)

        # 9. Relative Goal Direction Vector HUD
        hud_vector_y = 370
        vec_title = self.font_subtitle.render("Relative Goal Dir (2D)", True, self.COLOR_GOAL)
        self.screen.blit(vec_title, (self.width - self.panel_width + 25, hud_vector_y))
        
        dx_val, dy_val = obs[9], obs[10]
        vec_str = f"[{dx_val:+.3f}, {dy_val:+.3f}]"
        vec_surf = self.font_mono.render(vec_str, True, self.COLOR_TEXT)
        self.screen.blit(vec_surf, (self.width - self.panel_width + 25, hud_vector_y + 24))

        # Compass Visualizer
        compass_center = (self.width - self.panel_width + 200, hud_vector_y + 50)
        compass_r = 25
        pygame.draw.circle(self.screen, self.COLOR_GRID, compass_center, compass_r, 1)
        pygame.draw.circle(self.screen, self.COLOR_GRID_INNER, compass_center, 3)
        
        if not (dx_val == 0.0 and dy_val == 0.0):
            arrow_end = (
                int(compass_center[0] + dx_val * compass_r),
                int(compass_center[1] + dy_val * compass_r)
            )
            pygame.draw.line(self.screen, self.COLOR_GOAL, compass_center, arrow_end, 2)
            pygame.draw.circle(self.screen, self.COLOR_GOAL, arrow_end, 3)

        # 10. Observation Vector Output (11-dim)
        hud_obs_y = 470
        obs_title = self.font_subtitle.render("Observation State (11-dim)", True, self.COLOR_TEXT)
        self.screen.blit(obs_title, (self.width - self.panel_width + 25, hud_obs_y))
        
        # Print actual 11 float values nicely formatted
        obs_str1 = " ".join([f"{v:.1f}" for v in obs[:9]])
        obs_str2 = f"[{obs[9]:.2f}, {obs[10]:.2f}]"
        
        obs_surf1 = self.font_mono.render(f"FOV:  {obs_str1}", True, self.COLOR_TEXT_DIM)
        obs_surf2 = self.font_mono.render(f"Goal: {obs_str2}", True, self.COLOR_TEXT_DIM)
        self.screen.blit(obs_surf1, (self.width - self.panel_width + 25, hud_obs_y + 24))
        self.screen.blit(obs_surf2, (self.width - self.panel_width + 25, hud_obs_y + 42))

        # 11. Draw Neural Activity Heatmap (if SNN data available)
        self._draw_neural_heatmap()

    # ------------------------------------------------------------------
    # SNN Visualization helpers
    # ------------------------------------------------------------------

    def _draw_grid_overlay(self):
        """Render motor spike count and current action on the grid area (semi-transparent)."""
        if self.snn_data is None:
            return

        d = self.snn_data
        motor_total = d.get('motor_total', 0)
        action_name = d.get('action_name', '—')
        step = d.get('step', 0)

        # Build overlay surface
        overlay_w, overlay_h = 220, 50
        overlay = pygame.Surface((overlay_w, overlay_h), pygame.SRCALPHA)
        overlay.fill(self.COLOR_OVERLAY_BG)

        # Render text onto overlay
        line1 = f"Step {step:03d}  MOTOR: {motor_total}"
        line2 = f"Action: {action_name}"
        surf1 = self.font_mono.render(line1, True, self.COLOR_TEXT)
        surf2 = self.font_mono.render(line2, True, self.COLOR_FOV)
        overlay.blit(surf1, (8, 6))
        overlay.blit(surf2, (8, 26))

        # Blit onto screen at top-left of grid area
        self.screen.blit(overlay, (self.grid_offset_x + 4, self.grid_offset_y + 4))

    def _draw_neural_heatmap(self):
        """Draw the neuron firing heatmap in the side panel, color-coded by population."""
        if self.snn_data is None:
            return

        d = self.snn_data
        sensory = np.asarray(d.get('sensory_spikes', []))
        recurrent = np.asarray(d.get('recurrent_spikes', []))
        motor = np.asarray(d.get('motor_spikes', []))

        if len(sensory) == 0 and len(recurrent) == 0 and len(motor) == 0:
            return

        panel_x = self.width - self.panel_width + 25
        heatmap_y = 530  # start below existing HUD elements

        # Section title
        title = self.font_subtitle.render("Neural Activity Heatmap", True, self.COLOR_AGENT)
        self.screen.blit(title, (panel_x, heatmap_y))
        heatmap_y += 22

        cell_px = 3   # pixel size per neuron cell
        gap = 4       # gap between population blocks

        # Draw each population as a 10-column grid
        cols = 10
        populations = [
            ("Sensory",   sensory,   self.COLOR_SENSORY,   100),
            ("Recurrent", recurrent, self.COLOR_RECURRENT, 800),
            ("Motor",     motor,     self.COLOR_MOTOR,     100),
        ]

        for pop_name, spikes, color, expected_size in populations:
            if len(spikes) != expected_size:
                continue

            rows = expected_size // cols
            block_w = cols * cell_px
            block_h = rows * cell_px

            # Population label
            label = self.font_mono.render(pop_name, True, color)
            self.screen.blit(label, (panel_x, heatmap_y))
            heatmap_y += 16

            # Draw neuron grid
            max_spike = max(float(np.max(spikes)), 1.0)
            for idx in range(expected_size):
                r = idx // cols
                c = idx % cols
                x = panel_x + c * cell_px
                y = heatmap_y + r * cell_px

                if spikes[idx] > 0:
                    # Brightness proportional to spike count
                    intensity = min(spikes[idx] / max_spike, 1.0)
                    # Lerp from dim to full color
                    cr = int(self.COLOR_NEURON_DIM[0] + (color[0] - self.COLOR_NEURON_DIM[0]) * intensity)
                    cg = int(self.COLOR_NEURON_DIM[1] + (color[1] - self.COLOR_NEURON_DIM[1]) * intensity)
                    cb = int(self.COLOR_NEURON_DIM[2] + (color[2] - self.COLOR_NEURON_DIM[2]) * intensity)
                    cell_color = (cr, cg, cb)
                else:
                    cell_color = self.COLOR_NEURON_DIM

                pygame.draw.rect(self.screen, cell_color, (x, y, cell_px, cell_px))

            # Draw border around the block
            pygame.draw.rect(self.screen, self.COLOR_GRID, (panel_x, heatmap_y, block_w, block_h), 1)
            heatmap_y += block_h + gap

        # SNN summary stats below heatmap
        heatmap_y += 4
        stats = [
            ("Motor Spikes:", f"{d.get('motor_total', 0)}"),
            ("Firing %:",     f"{d.get('motor_firing_pct', 0.0):.1f}%"),
            ("Free Energy:",  f"{d.get('free_energy', 0.0):.3f}"),
            ("Action:",       f"{d.get('action_name', '—')}"),
        ]
        for label, val in stats:
            lbl_surf = self.font_body.render(label, True, self.COLOR_TEXT_DIM)
            val_surf = self.font_mono.render(val, True, self.COLOR_TEXT)
            self.screen.blit(lbl_surf, (panel_x, heatmap_y))
            self.screen.blit(val_surf, (panel_x + 120, heatmap_y))
            heatmap_y += 20
