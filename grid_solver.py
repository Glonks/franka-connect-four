import numpy as np

class GridSolver:
    grid1 = np.array([[1, 1, 1], [0, 2, 0], [0, 2, 0]])
    grid2 = np.array([[1, 1, 0], [1, 2, 0], [0, 2, 0]])
    grid3 = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]])
    grid4 = np.array([[1, 2, 0], [1, 0, 2], [1, 2, 0]])
    colors = {1 : "blue", 2 : "red"}

    def _solve(self, goal, current_sol):
        # Fill in current state
        if len(current_sol) == np.sum(goal > 0):
            return current_sol
        state = np.zeros_like(goal)
        for (i, j, _) in current_sol:
            state[i, j] = 1
        
        # Try each possible next move
        for i in range(3):
            for j in range(3):
                if goal[i, j] == 0 or state[i, j] == 1:
                    continue
                
                # Try vertical grasps
                if (i == 0 or state[i-1, j] == 0) and (i == 2 or state[i+1, j] == 0):
                    res = self._solve(goal, current_sol + [(i, j, 'v')])
                    if res is not None:
                        return res

                if (j == 0 or state[i, j-1] == 0) and (j == 2 or state[i, j+1] == 0):
                    res = self._solve(goal, current_sol + [(i, j, 'h')])
                    if res is not None:
                        return res
            
        return None
    
    def plan(self):
        output = []
        for grid in [self.grid1, self.grid2, self.grid3, self.grid4]:
            soln = self._solve(grid, [])

            if soln is not None:
                plan_leg_f = [("place", self.colors[grid[i, j]], (i, j), d) for (i, j, d) in soln]
                plan_leg_b = [("remove", color, coord, d) for (_, color, coord, d) in reversed(plan_leg_f)]
                output += plan_leg_f + plan_leg_b
        
        return output
    

solver = GridSolver()
print(solver.plan())