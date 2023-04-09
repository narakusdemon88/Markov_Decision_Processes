import hiive.mdptoolbox
from jons_hiive.hiive.mdptoolbox.example import openai
from sklearn.model_selection import GridSearchCV
from gymnasium.envs.toy_text import frozen_lake
from jons_hiive.hiive.mdptoolbox import mdp

class Policy:
    def __init__(self, policy, map_size, probability):
        self.policy = policy
        self.frozen_map = self.generate_map(map_size=map_size, probability=probability)
        self.color_map = {
            "S": "orange",  # Start
            "G": "green",  # Goal
            "F": "xkcd:sky blue",  # Free spot
            "H": "aquamarine"  # Hole
        }
        self.directions = {
            0: "L",
            1: "D",
            2: "R",
            3: "U",
            "fin": "#"
        }
        # self.figure = plt.figure()
        # self.subplot = self.figure.add_subplot(xlim=(0, map_size), ylim=(0, map_size))

    @staticmethod
    def generate_map(map_size, probability):
        return frozen_lake.generate_random_map(
            size=map_size,
            p=probability,
            seed=909
        )


frozen_map = Policy.generate_map(map_size=5, probability=0.9)

P, R = openai(env_name="FrozenLake-v1", desc=frozen_map)

param_grid = {
    'gamma': [0.9, 0.95, 0.99],
    'epsilon': [0.1, 0.2, 0.3],
    'alpha': [0.1, 0.2, 0.3],
    'tol': [0.0001, 0.00001, 0.000001]
}
ql = mdp.QLearning(P, R, gamma=0.99, alpha=0.1, epsilon=0.1, n_iter=10000)
grid_search = GridSearchCV(ql, param_grid, cv=5)
grid_search.fit(P, R)
print("Best parameters: ", grid_search.best_params_)
print("Best mean test score: ", grid_search.best_score_)
