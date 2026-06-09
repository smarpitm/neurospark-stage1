import numpy as np

class GridWorld:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self, grid_size=10, start_pos=(0, 0), goal_pos=(9, 9)):
        """
        Initializes a 10x10 Grid World.
        :param grid_size: Dimensions of the grid (default 10)
        :param start_pos: Starting position of the agent (default (0,0))
        :param goal_pos: Goal position (default (9,9))
        """
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.reset()

    def reset(self):
        """
        Resets the environment.
        :return: Initial 11-dimensional observation vector.
        """
        self.agent_pos = list(self.start_pos)  # [x, y]
        self.done = False
        return self.get_observation()

    def get_actions(self):
        """
        Returns list of valid actions.
        """
        return [self.UP, self.DOWN, self.LEFT, self.RIGHT]

    def get_observation(self):
        """
        Generates the 11-dimensional observation vector:
        - 3x3 local patch around agent (9 values): 1.0 for wall/out-of-bounds, 0.0 for empty space.
        - Relative goal direction (2 values): [dx, dy] normalized to unit length, or [0, 0] if at goal.
        """
        patch = []
        ax, ay = self.agent_pos
        
        # 3x3 local patch in row-major order:
        # y-1: x-1, x, x+1
        # y:   x-1, x, x+1
        # y+1: x-1, x, x+1
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                x = ax + dx
                y = ay + dy
                if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                    patch.append(1.0)  # Wall / Boundary
                else:
                    patch.append(0.0)  # Free space
        
        # Relative goal direction (2 values)
        dx = float(self.goal_pos[0] - ax)
        dy = float(self.goal_pos[1] - ay)
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            rel_dir = [dx / dist, dy / dist]
        else:
            rel_dir = [0.0, 0.0]

        # Concatenate to get 11-dimensional vector
        obs = np.array(patch + rel_dir, dtype=np.float32)
        return obs

    def step(self, action):
        """
        Executes one step in the environment.
        :param action: Action index (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT)
        :return: (next_state, reward, done)
        """
        if self.done:
            return self.get_observation(), 0.0, True

        ax, ay = self.agent_pos
        next_x, next_y = ax, ay

        # Apply action
        if action == self.UP:
            next_y = ay - 1
        elif action == self.DOWN:
            next_y = ay + 1
        elif action == self.LEFT:
            next_x = ax - 1
        elif action == self.RIGHT:
            next_x = ax + 1
        else:
            raise ValueError(f"Invalid action: {action}")

        # Collision detection
        collision = (next_x < 0 or next_x >= self.grid_size or 
                     next_y < 0 or next_y >= self.grid_size)

        if collision:
            reward = -1.0
            # Position remains the same
        else:
            self.agent_pos = [next_x, next_y]
            if self.agent_pos == list(self.goal_pos):
                reward = 1.0
                self.done = True
            else:
                reward = -0.01

        return self.get_observation(), reward, self.done

if __name__ == "__main__":
    # If run directly, import the renderer and run the interactive/random loop.
    import time
    from renderer import Renderer
    
    env = GridWorld()
    renderer = Renderer(env)
    
    print("Starting Grid World!")
    print("Controls: Arrow keys to move agent. Press ESC to quit.")
    print("If no keys are pressed, agent will move randomly after a short delay.")
    
    renderer.run()
