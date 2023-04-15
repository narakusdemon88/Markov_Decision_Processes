"""
Frozen Lake Policy Iteration
Jon-Erik Akashi jakashi3@gatech.edu
"""
# TODO: compare to original, save files, increase ranges?
import time
from gymnasium.envs.toy_text import frozen_lake
from jons_hiive.hiive.mdptoolbox.example import openai
from jons_hiive.hiive.mdptoolbox import mdp
import pandas as pd
import matplotlib.pyplot as plt


def get_probability_reward(environment_name, map):
    probabilities, rewards = openai(
        env_name=environment_name,
        desc=map
    )
    return probabilities, rewards


def policy_iteration(gamma_range: list, probabilities, rewards):
    all_policies = []

    for gamma in gamma_range:
        print(gamma)
        one_policy = {}
        t1 = time.perf_counter()
        policy_iteration = mdp.PolicyIterationModified(
            transitions=probabilities,
            reward=rewards,
            gamma=float(gamma),
            epsilon=0.0001,  # set to 0 for greedy
            max_iter=10_000
        )
        policy_iteration.run()
        t2 = time.perf_counter()

        # get results
        wall_clock_time = t2 - t1
        score = float(sum(policy_iteration.V) / len(policy_iteration.V))
        policy = policy_iteration.policy

        # append results to list
        one_policy["iteration"] = policy_iteration.iter
        one_policy["time"] = wall_clock_time
        one_policy["score"] = score
        one_policy["policy"] = policy

        all_policies.append(one_policy)
    return all_policies


def calculate_rewards(probabilities, rewards):
    main_policy = mdp.PolicyIteration(
        transitions=probabilities,
        reward=rewards,
        gamma=0.999,
    )
    return pd.DataFrame(main_policy.run())["Max V"]


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
        self.figure = plt.figure()
        self.subplot = self.figure.add_subplot(xlim=(0, map_size), ylim=(0, map_size))

    @staticmethod
    def generate_map(map_size, probability):
        return frozen_lake.generate_random_map(
            size=map_size,
            p=probability,
            seed=909
        )


def plot_results(x_values, y_values, ylabel, title):
    plt.plot(x_values, y_values)
    plt.xlabel("Discount Value")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(visible=True)
    plt.show()
    plt.clf()


def plot_initial(map_size: int, plot, last_policy):
    for row in range(map_size):
        for col in range(map_size):
            y = map_size - row - 1
            state = plt.Rectangle(xy=(col, y), height=1.0, width=1.0)
            state.set_facecolor(plot.color_map[plot.frozen_map[row][col]])
            plot.subplot.add_patch(state)
            if row == map_size - 1 and col == map_size - 1 or plot.color_map[plot.frozen_map[row][col]] == "aquamarine":
                continue
            plot.subplot.text(col + .5, y + .5, plot.directions[last_policy[row * map_size + col]],
                              horizontalalignment="center",
                              verticalalignment="center",
                              color="white")
    plt.title(f"Value Iteration {map_size}x{map_size} Frozen Lake Grid")
    plt.show()
    plt.clf()


def plot_convergence(rolling):
    iterations = [i + 1 for i in range(len(rolling))]
    plt.plot(iterations, rolling)
    plt.xlabel("Iterations")
    plt.ylabel("Reward Delta")
    plt.title(f"Convergence (PI) Frozen Lake")
    plt.tight_layout()
    plt.grid(visible=True)
    plt.show()
    plt.clf()


def main():
    # Set up main variables
    map_size = 25
    initial_probability = 0.9
    env_name = "FrozenLake-v1"
    gamma_range = [i / 10 for i in range(1, 10)]

    # Create the frozen lake map
    frozen_map = Policy.generate_map(map_size=map_size, probability=initial_probability)

    # Calculate the probability and reward
    probabilities, rewards = get_probability_reward(environment_name=env_name, map=frozen_map)

    all_policies = policy_iteration(gamma_range=gamma_range, probabilities=probabilities, rewards=rewards)

    rolling = calculate_rewards(probabilities=probabilities, rewards=rewards)

    last_policy = all_policies[-1]["policy"]

    # Plot policy
    plot_initial(map_size, Policy(policy=last_policy, map_size=map_size, probability=initial_probability), last_policy)

    # Plot Metrics
    # plot_results(x_values=gamma_range, y_values=[i["iteration"] for i in all_policies], ylabel="Iterations",
    #              title="Gamma & Iteration Values (PI) Frozen Lake")
    # plot_results(x_values=gamma_range, y_values=[i["time"] for i in all_policies], ylabel="Time (Seconds)",
    #              title="Gamma & Wall Clock Times (PI) Frozen Lake")
    plot_results(x_values=gamma_range, y_values=[i["score"]*500 for i in all_policies], ylabel="Score (Avg)",  # TODO: REMOVE THE 500
                 title="Gamma & Score (PI) Frozen Lake")

    # Plot Convergence
    plot_convergence(rolling=[i*500 for i in rolling])  # TODO: REMOVE THE 500


if __name__ == "__main__":
    main()
