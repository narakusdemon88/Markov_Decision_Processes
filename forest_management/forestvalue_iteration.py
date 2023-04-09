"""
Value Iteration for Forest Management
Jon-Erik Akashi (jakashi3@gatech.edu)
"""
# TODO: change gamma_range var
import pandas as pd
from hiive.mdptoolbox import example
from hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning
import time
import matplotlib.pyplot as plt
import numpy as np


def instantiate_prob_rewards(map_size, r1, r2, start_prob):
    prob, reward = example.forest(
        S=map_size,
        r1=r1,
        r2=r2,
        p=start_prob
    )
    return prob, reward


def value_iteration(probabilities, rewards, gamma_range: list, size):
    all_iters = []

    for gamma in gamma_range:
        one_iter = {}
        t1 = time.perf_counter()
        pol_iter = ValueIteration(
            transitions=probabilities,
            reward=rewards,
            max_iter=10_000,
            gamma=gamma
        )
        pol_iter.run()
        t2 = time.perf_counter()

        # add stuff to dictionary
        one_iter["time"] = t2 - t1
        one_iter["iterations"] = pol_iter.iter
        one_iter["policy"] = pol_iter.policy
        one_iter["score"] = max(pol_iter.V)

        all_iters.append(one_iter)

    last_iter_val = [i["iterations"] for i in all_iters][-1] * 2

    converge_list = []
    # for i in range(1, size):
    for i in np.linspace(1, last_iter_val, 20):
        poly = ValueIteration(
            transitions=probabilities,
            reward=rewards,
            gamma=0.999,
            max_iter=i
        )
        converge_list.append(pd.DataFrame(poly.run())["Mean V"].iloc[-1])

    return all_iters, converge_list


def plot_results(x_values, y_values, ylabel, title):
    plt.plot(x_values, y_values)
    plt.xlabel("Discount Value")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(visible=True)
    plt.show()
    plt.clf()


def calculate_rewards(probabilities, rewards):
    main_policy = ValueIteration(
        transitions=probabilities,
        reward=rewards,
        gamma=.999,
        epsilon=.0001,
        max_iter=10_000
    )
    # res = main_policy.run()
    #
    # errors = [0]
    #
    # errors = errors + [i["Error"] for i in res]
    # deltas = []
    # for i in range(len(errors) - 1):
    #     delta = errors[i + 1] - errors[i]
    #     deltas.append(abs(delta))

    return pd.DataFrame(main_policy.run())["Max V"]


def plot_convergence(rolling):
    iterations = [i + 1 for i in range(len(rolling))]
    plt.plot(iterations, rolling)
    plt.xlabel("Iterations")
    plt.ylabel("Reward Delta")
    plt.title(f"Convergence (VI) Forest")
    plt.tight_layout()
    plt.grid(visible=True)
    plt.show()
    plt.clf()


def main():
    # Set basic values
    map_size = 50
    r1 = 50
    r2 = 2
    start_prob = 0.1
    # gamma_range = [.0001, .001, .01, .1, .3, .6, .8, .9, .95, .999]  # TODO: change this
    gamma_range = [i / 10 for i in range(1, 10)]

    probabilities, rewards = instantiate_prob_rewards(
        map_size=map_size,
        r1=r1,
        r2=r2,
        start_prob=start_prob)

    all_iters, converge_list = value_iteration(probabilities, rewards, gamma_range, map_size)

    rolling = calculate_rewards(probabilities=probabilities, rewards=rewards)

    plot_results(x_values=gamma_range, y_values=[i["iterations"] for i in all_iters], ylabel="Iterations",
                 title="Gamma & Iteration Values (VI) Forest")
    plot_results(x_values=gamma_range, y_values=[i["time"] for i in all_iters], ylabel="Time (Seconds)",
                 title="Gamma & Wall Clock Times (VI) Forest")
    plot_results(x_values=gamma_range, y_values=[i["score"]*10 for i in all_iters], ylabel="Score (Avg)",  # TODO: REMOVE THE 10
                 title="Gamma & Score (VI) Forest")
    plot_convergence(rolling=[i*10 for i in rolling])  # TODO: REMOVE THE 500


if __name__ == "__main__":
    main()
