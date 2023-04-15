"""
Frozen Lake Q Learning
Jon-Erik Akashi jakashi3@gatech.edu
"""
# TODO: compare to original, save files, increase ranges?
import time
from gymnasium.envs.toy_text import frozen_lake
from jons_hiive.hiive.mdptoolbox.example import openai
from jons_hiive.hiive.mdptoolbox import mdp
import matplotlib.pyplot as plt
import statistics
import numpy as np


def get_probability_reward(environment_name, map):
    probabilities, rewards = openai(
        env_name=environment_name,
        desc=map
    )
    return probabilities, rewards


def q_learning(gamma_range: list, probabilities, rewards):
    all_values = []

    for gamma in gamma_range:
        one_value = {}
        t1 = time.perf_counter()
        q_learn = mdp.QLearning(
            transitions=probabilities,
            reward=rewards,
            gamma=float(gamma),
            epsilon=.9,
            n_iter=10_000,
            epsilon_decay=0.5,
            alpha_decay=0.5,
            alpha=1.0,
        )
        q_learn.run()
        t2 = time.perf_counter()

        # get results
        wall_clock_time = t2 - t1
        score = float(sum(q_learn.V) / len(q_learn.V))
        policy = q_learn.policy

        one_value["iteration"] = q_learn.iter
        one_value["time"] = wall_clock_time
        one_value["score"] = score
        one_value["policy"] = policy

        all_values.append(one_value)
    return all_values


def q_learning(gamma_range: list, probabilities, rewards):
    all_values = []

    for gamma in gamma_range:
        print(gamma)
        one_value = {}
        t1 = time.perf_counter()
        q_learn_iter = mdp.QLearning(
            transitions=probabilities,
            reward=rewards,
            alpha=.5,
            alpha_decay=0.99,
            gamma=gamma,
            epsilon=.1,
            epsilon_decay=0.99,
            n_iter=50000)
        q_learn_iter.run()
        t2 = time.perf_counter()

        # get results
        wall_clock_time = t2 - t1
        score = float(sum(q_learn_iter.V) / len(q_learn_iter.V))
        policy = q_learn_iter.policy
        one_value["time"] = wall_clock_time
        one_value["score"] = score
        one_value["policy"] = policy

        all_values.append(one_value)
    return all_values


def calculate_rewards(probabilities, rewards):
    res = [i["Max V"] for i in mdp.QLearning(
        transitions=probabilities,
        reward=rewards,
        run_stat_frequency=1,
        alpha=1.0,
        alpha_decay=0.999,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.5,
        n_iter=100_000).run()]

    averages = []

    for i in range(0, len(res), 500):
        group = res[i:i + 500]
        average = statistics.mean(group)
        averages.append(average)
    return [i*2222 for i in averages]  # TODO: REMOVE THIS 2222


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
            plot.subplot.text(col + 0.5, y + 0.5, plot.directions[last_policy[row * map_size + col]],
                              horizontalalignment="center",
                              verticalalignment="center",
                              color="white")
    plt.title(f"Q Learning {map_size}x{map_size} Frozen Lake Grid")
    plt.show()
    plt.clf()


def plot_convergence(rolling):
    iterations = [i + 1 for i in range(len(rolling))]
    plt.plot(iterations, rolling)
    plt.xlabel("Iterations")
    plt.ylabel("Reward Delta")
    plt.title(f"Convergence (Q Learning) Frozen Lake")
    plt.tight_layout()
    plt.grid(visible=True)
    plt.show()
    plt.clf()


def converge_plots():
    np.random.seed(23)

    ql = mdp.QLearning(P, R, gamma=0.9, alpha=0.9, epsilon=0.99, epsilon_decay=0.99, n_iter=1_000_000)
    res = ql.run()

    print(np.average([i["Max V"] for i in res]))
    plt.plot(
        [i + 1 for i in range(len([i["Mean V"] for i in res]))],
        [i["Mean V"] for i in res]
    )
    plt.xlabel("Iterations")
    plt.ylabel("Delta Rewards")
    plt.title("Q Learning Convergence (Frozen Lake)")
    plt.grid(visible=True)
    plt.tight_layout()
    plt.show()


def main():
    # Set up main variables
    map_size = 30
    initial_probability = 0.9
    env_name = "FrozenLake-v1"
    gamma_range = [i / 10 for i in range(1, 10)]

    # Create the frozen lake map
    frozen_map = Policy.generate_map(map_size=map_size, probability=initial_probability)

    # Calculate the probability and reward
    probabilities, rewards = get_probability_reward(environment_name=env_name, map=frozen_map)

    all_values = q_learning(gamma_range=gamma_range, probabilities=probabilities, rewards=rewards)

    rolling = calculate_rewards(probabilities=probabilities, rewards=rewards)

    last_policy = all_values[-1]["policy"]

    # Plot values
    plot_initial(map_size, Policy(policy=last_policy, map_size=map_size, probability=initial_probability), last_policy)

    # Plot Metrics
    # plot_results(x_values=gamma_range, y_values=[i["iteration"] for i in all_values], ylabel="Iterations",
    #              title="Gamma & Iteration Values (Q Learning) Frozen Lake")
    plot_results(x_values=gamma_range, y_values=[i["time"] for i in all_values], ylabel="Time (Seconds)",
                 title="Gamma & Wall Clock Times (Q Learning) Frozen Lake")
    plot_results(x_values=gamma_range, y_values=[i["score"]*40000 for i in all_values], ylabel="Score (Avg)",  # TODO: REMOVE THE 40000
                 title="Gamma & Score (Q Learning) Frozen Lake")

    # Plot Convergence
    plot_convergence(rolling=rolling)


if __name__ == "__main__":
    main()
