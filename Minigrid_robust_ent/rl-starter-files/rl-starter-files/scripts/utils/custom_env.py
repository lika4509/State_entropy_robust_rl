from collections import deque

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall,Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from gym.envs.registration import register
import numpy as np
import gym
import pygame
import matplotlib.pyplot as plt
class EmptyEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps= 200,
        blocked=False,
        num_blocks=15,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.grid_size = size

        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        #declare the action space
        self.action_space = gym.spaces.Discrete(3)
        #declare the observation space
        self.blocked = blocked
        self.num_blocks = int(num_blocks)
        self.observation_space = gym.spaces.Dict({'image': gym.spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype='uint8')})
        # Layouts are generated from a per-environment seed and a reset counter.
        # This keeps them independent of the global NumPy RNG and reproducible
        # across policies that complete the same number of episodes.
        self._layout_seed = None
        self._layout_counter = 0
        self.occupancy = np.zeros((self.grid_size, self.grid_size))
        self.possible_block_modes =[False,1,2,3,4,5,6,7,8,9]

        


    @staticmethod
    def _gen_mission():
        return "reach the goal"
    def seed(self, seed=None):
        # Store the layout seed for this environment instead of seeding the
        # global NumPy RNG.
        self._layout_seed = seed
        self._layout_counter = 0
        return [seed]

    @staticmethod
    def _is_solvable(blocked_cells, start, goal, width, height):
        """Return True if a 4-connected path exists around blocked cells."""
        if start == goal:
            return True

        queue = deque([start])
        visited = {start}
        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (1 <= nx < width - 1 and 1 <= ny < height - 1):
                    continue
                if (nx, ny) in blocked_cells or (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        return False

    def _random_wall_cells(self, rng, width, height, goal):
        """Sample a random straight wall, either partial or full with one gap."""
        start = self.agent_start_pos or (1, 1)

        for _ in range(100):
            vertical = rng.rand() < 0.5
            full = rng.rand() < 0.5
            cells = set()

            if vertical:
                x = rng.randint(3, width - 3)
                if full:
                    gap_y = rng.randint(2, height - 2)
                    cells = {
                        (x, y) for y in range(1, height - 1) if y != gap_y
                    }
                else:
                    length = rng.randint(5, min(7, height - 3) + 1)
                    y0 = rng.randint(1, height - length)
                    cells = {(x, y0 + i) for i in range(length)}
            else:
                y = rng.randint(3, height - 3)
                if full:
                    gap_x = rng.randint(2, width - 2)
                    cells = {
                        (x, y) for x in range(1, width - 1) if x != gap_x
                    }
                else:
                    length = rng.randint(5, min(7, width - 3) + 1)
                    x0 = rng.randint(1, width - length)
                    cells = {(x0 + i, y) for i in range(length)}

            if self._is_solvable(cells, start, goal, width, height):
                return cells

        return set()

    def _dense_block_cells(self, rng, width, height, num_blocks, goal):
        """Sample random block cells and reject layouts that block the goal."""
        start = self.agent_start_pos or (1, 1)
        allowed = [
            (x, y)
            for x in range(1, width - 1)
            for y in range(1, height - 1)
            if (x, y) != start and (x, y) != goal
        ]
        num_blocks = min(int(num_blocks), len(allowed))

        for _ in range(100):
            indices = rng.choice(len(allowed), num_blocks, replace=False)
            cells = {allowed[i] for i in indices}
            if self._is_solvable(cells, start, goal, width, height):
                return cells

        # Fallback: add blocks one at a time while the goal remains reachable.
        cells = set()
        for i in rng.permutation(len(allowed)):
            if len(cells) >= num_blocks:
                break
            candidate = cells | {allowed[i]}
            if self._is_solvable(candidate, start, goal, width, height):
                cells = candidate
        return cells

    def _corner_goal(self, rng, width, height):
        if rng.rand() < 0.5:
            return width - 2, 1
        return 1, height - 2

    def _rooms_cells(self, rng, width, height, goal):
        """Sample a 2x2 room layout with one gap in each interior wall."""
        start = self.agent_start_pos or (1, 1)
        x_wall = width // 2
        y_wall = height // 2

        for _ in range(100):
            # There are two solvable room paths:
            #   top-left -> top-right -> bottom-right
            #   top-left -> bottom-left -> bottom-right
            if rng.rand() < 0.5:
                gap_y = rng.randint(2, y_wall)          # top half
                gap_x = rng.randint(x_wall + 1, width - 2)  # right half
            else:
                gap_y = rng.randint(y_wall + 1, height - 2)  # bottom half
                gap_x = rng.randint(2, x_wall)               # left half

            cells = set()
            for y in range(1, height - 1):
                if y != gap_y:
                    cells.add((x_wall, y))
            for x in range(1, width - 1):
                if x != gap_x:
                    cells.add((x, y_wall))

            if self._is_solvable(cells, start, goal, width, height):
                return cells

        return set()

    def _maze_cells(self, rng, width, height):
        """Sample a small 1-cell-corridor maze with a reachable goal."""
        start = self.agent_start_pos or (1, 1)
        goal = (width - 2, height - 2)
        cells = {
            (x, y)
            for x in range(1, width - 1)
            for y in range(1, height - 1)
        }
        maze_cells = [
            (x, y)
            for x in range(1, width - 1, 2)
            for y in range(1, height - 1, 2)
        ]
        if start not in maze_cells:
            return set()

        visited = {start}
        stack = [start]
        cells.discard(start)

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in maze_cells and (nx, ny) not in visited:
                    neighbors.append((nx, ny, dx, dy))
            if not neighbors:
                stack.pop()
                continue

            nx, ny, dx, dy = neighbors[rng.randint(0, len(neighbors))]
            cells.discard((x + dx // 2, y + dy // 2))
            cells.discard((nx, ny))
            visited.add((nx, ny))
            stack.append((nx, ny))

        # Connect the far corner to the odd-cell maze.
        if rng.rand() < 0.5:
            connector = (width - 2, height - 3)
        else:
            connector = (width - 3, height - 2)
        cells.discard(connector)
        cells.discard(goal)

        if self._is_solvable(cells, start, goal, width, height):
            return cells
        return set()

    def _gen_grid(self, width, height):
        # Use a per-environment RNG seeded by the environment seed and the
        # number of layouts generated so far.
        base_seed = self._layout_seed
        if base_seed is None:
            base_seed = np.random.SeedSequence().entropy
        layout_seed = np.random.SeedSequence(
            [int(base_seed), self._layout_counter]
        ).generate_state(1)[0]
        rng = np.random.RandomState(layout_seed)
        self._layout_counter += 1

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        
        if self.blocked == 1:
             x_blocked=4
             blocks= [(x_blocked,5),(x_blocked,4),(x_blocked,3),(x_blocked,2),(x_blocked,1)]
             for i in range(len(blocks)):
                self.grid.set(blocks[i][0],blocks[i][1],Wall())

        elif self.blocked == 2:
            y_blocked = rng.randint(1, height-5)
            blocks =[(5,y_blocked),(6,y_blocked),(7,y_blocked),(5,y_blocked+1),(5,y_blocked+2)]
            for i in range(len(blocks)):
                self.grid.set(blocks[i][0],blocks[i][1],Wall())
            
        elif self.blocked == 3:
            all_points = [[x, y] for x in range(2, width-1) for y in range(2, height-1)]
            #remove the points (2,2),(8,8):
            all_points.remove([2,2])
            all_points.remove([8,8])
            all_points = np.array(all_points)
            
            num_points = 7
            random_indices = rng.choice(all_points.shape[0], num_points, replace=False)
            random_points = all_points[random_indices]
            for i in range(len(random_points)):
                self.grid.set(random_points[i][0],random_points[i][1],Wall())

        elif self.blocked == 5:
            # Random straight wall: partial, or full with exactly one gap.
            goal = (width - 2, height - 2)
            for x, y in self._random_wall_cells(rng, width, height, goal):
                self.grid.set(x, y, Wall())

        elif self.blocked == 6:
            # Dense random blocks, rejection-sampled until the goal is reachable.
            goal = (width - 2, height - 2)
            blocks = self._dense_block_cells(
                rng, width, height, self.num_blocks, goal
            )
            for x, y in blocks:
                self.grid.set(x, y, Wall())

        elif self.blocked == 8:
            # 2x2 rooms connected by one gap in each interior wall.
            goal = (width - 2, height - 2)
            for x, y in self._rooms_cells(rng, width, height, goal):
                self.grid.set(x, y, Wall())

        elif self.blocked == 9:
            # Small 1-cell-corridor maze with a reachable goal.
            goal = (width - 2, height - 2)
            for x, y in self._maze_cells(rng, width, height):
                self.grid.set(x, y, Wall())

        # Place a goal square in the bottom-right corner
        if self.blocked == 4:
            # generate a random position for the goal
            x_goal = rng.randint(2, width-2)
            y_goal = rng.randint(2, height-2)
            # set the goal position
            self.put_obj(Goal(), x_goal, y_goal)

        elif self.blocked == 7:
            # Extreme reward shift: one goal in a non-training corner.
            x_goal, y_goal = self._corner_goal(rng, width, height)
            self.put_obj(Goal(), x_goal, y_goal)

        else:
            self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "reach the goal"
        
    #inherit the step function from the parent class
    def step(self, action):
        obs, reward, done,truncated, info = super().step(action)

        if isinstance(self.grid.get(*self.agent_pos), self.PenaltyTile):
            #with probability 0.75 a penalty is given:
            if self.np_random.random() < 0.75:
                reward += -0.2
            #if steps limit isnt met, then the agent isnt done
            if self.step_count < self.max_steps:
                done = False
        #mask unwanted actions
        # if action in [3,4,5,6]:
        #     reward += -10

        if done or truncated:
            done = True
        self.occupancy[self.agent_pos[0], self.agent_pos[1]] += 1
        info = dict(info)
        info["success"] = isinstance(self.grid.get(*self.agent_pos), Goal)

        return obs, reward, done, info
    
    def reset(self,seed=None):
        if seed is not None:
            self._layout_seed = seed
            self._layout_counter = 0
        self.occupancy = np.zeros((self.grid_size, self.grid_size))
        obs,_ = super().reset(seed=seed)
        return obs
    
    class PenaltyTile(Lava):
        def __init__(self):
            super().__init__()
        
        def render(self,img):
             super().render(img)
            
            
        def can_overlap(self):
            return True #allows the agent to step on the tile
    


def main():
    env = EmptyEnv(blocked= 4,render_mode='human')
    pygame.init()
    env.reset()
    while True:
        env.render()
    # enable manual control for testing
    # manual_control = ManualControl(env, seed=42,)
    
    # manual_control.start()
    # print(env.agent_dir)
    
    
    
if __name__ == "__main__":
    main()
#register the environment:
register(
    id='MiniGrid-EmptyEnv-v0',
    entry_point='utils.custom_env:EmptyEnv'
)
