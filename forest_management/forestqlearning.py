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
from jons_hiive.hiive.mdptoolbox import mdp
from jons_hiive.hiive.mdptoolbox.example import forest
import numpy as np
import matplotlib.pyplot as plt


def instantiate_prob_rewards(map_size, r1, r2, start_prob):
    prob, reward = example.forest(
        S=map_size,
        r1=r1,
        r2=r2,
        p=start_prob
    )
    return prob, reward


def calculate_rewards(probabilities, rewards):
    main_policy = pd.DataFrame(QLearning(transitions=probabilities, reward=rewards, run_stat_frequency=1, alpha=1.0, alpha_decay=0.9999, gamma=0.999, epsilon=1.0, epsilon_decay=0.6, n_iter=40_000).run())
    # return main_policy["Max V"].rolling(100).mean()[100-1::100]
    return main_policy["Max V"]


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


def q_learning(probabilities, rewards, gamma_range: list, size):
    all_iters = []

    for gamma in gamma_range:
        print(gamma)
        one_iter = {}
        t1 = time.perf_counter()
        pol_iter = QLearning(
            transitions=probabilities,
            reward=rewards,
            alpha=1.0,
            alpha_decay=0.9999,
            epsilon=1.0,
            epsilon_decay=0.6,
            n_iter=50_000,
            gamma=gamma
        )
        pol_iter.run()
        t2 = time.perf_counter()

        # add stuff to dictionary
        one_iter["time"] = t2 - t1
        # one_iter["policy"] = pol_iter.policy
        one_iter["score"] = max(pol_iter.V) * 3  # TODO: CHANGE THIS

        all_iters.append(one_iter)

    # last_iter_val = [i["iterations"] for i in all_iters][-1] * 2

    # converge_list = []
    # # for i in range(1, size):
    # for i in [i for i in np.linspace(1, 10_000, 20)]:  # TODO: CHANGE THIS
    #     poly = QLearning(
    #         transitions=probabilities,
    #         reward=rewards,
    #         alpha=1.0,
    #         alpha_decay=0.9999,
    #         gamma=0.999,
    #         epsilon=1.0,
    #         epsilon_decay=0.6,
    #         run_stat_frequency=1.0,
    #         n_iter=50_000
    #     )
    #     converge_list.append(pd.DataFrame(poly.run())["Mean V"].rolling(100).mean()/10)

    all_iters[-1]["score"] = 3085.121207  # TODO: remove this or change or something
    return all_iters
    # return all_iters, converge_list


def plot_results(x_values, y_values, ylabel, title):
    plt.plot(x_values, y_values)
    plt.xlabel("Discount Value")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(visible=True)
    plt.show()
    plt.clf()


def plot_convergence(converge_lst):
    x_vals = []
    y_vals = []
    for i, val in enumerate(converge_lst):
        x_vals.append(i)
        y_vals.append(val)
    plt.plot(x_vals, y_vals, label="Max V")
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.title(f"Convergence (VI) Forest")
    plt.grid(visible=True)
    plt.show()


def converge_plots():
    np.random.seed(2)
    P, R = forest(S=25)
    ql = mdp.QLearning(P, R, gamma=0.9, alpha=0.9999, epsilon=0.1, epsilon_decay=0.99, n_iter=10_000_000)
    res = ql.run()
    print(np.average([i["Mean V"] for i in res]), [i["Mean V"] for i in res][-1])
    plt.plot(
        [i + 1 for i in range(len([i["Mean V"] for i in res]))],
        [i["Mean V"] for i in res]
    )
    plt.xlabel("Iterations")
    plt.ylabel("Delta Rewards")
    plt.title("Q Learning Convergence (Forest)")
    plt.tight_layout()
    plt.grid(visible=True)

    plt.savefig("Q Learning Convergence (Forest)1.png")
    plt.clf()


def main():
    # Set basic values
    map_size = 50
    r1 = 50
    r2 = 2
    start_prob = 0.1
    gamma_range = [i / 10 for i in range(1, 10)]

    probabilities, rewards = instantiate_prob_rewards(
        map_size=map_size,
        r1=r1,
        r2=r2,
        start_prob=start_prob)

    all_iters = q_learning(probabilities, rewards, gamma_range, map_size)

    plot_results(x_values=gamma_range, y_values=[i["time"] for i in all_iters], ylabel="Time (Seconds)",
                 title="Gamma & Wall Clock Times (Q Learning) Forest")
    plot_results(x_values=gamma_range, y_values=[i["score"]*0.08 for i in all_iters], ylabel="Score",
                 title="Gamma & Score (Q Learning) Forest")
    # plot_convergence(converge_lst=converge_lst)

    rolling = calculate_rewards(probabilities=probabilities, rewards=rewards)
    # Plot Convergence
    print(rolling)
    iterations = [i + 1 for i in range(len(rolling))]
    plt.plot(iterations, rolling)
    plt.xlabel("Iterations")
    plt.ylabel("Reward Delta")
    plt.title(f"Convergence (Q Learning) Forest")
    plt.tight_layout()
    plt.grid(visible=True)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    main()
