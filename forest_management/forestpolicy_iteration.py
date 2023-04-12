"""
Policy Iteration for Forest Management
Jon-Erik Akashi (jakashi3@gatech.edu)
"""
# TODO: change gamma_range var
import pandas as pd
from hiive.mdptoolbox import example
from hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning
import time
import matplotlib.pyplot as plt


def instantiate_prob_rewards(map_size, r1, r2, start_prob):
    prob, reward = example.forest(
        S=map_size,
        r1=r1,
        r2=r2,
        p=start_prob
    )
    return prob, reward


def policy_iteration(probabilities, rewards, gamma_range: list, size):
    all_iters = []

    for gamma in gamma_range:
        one_iter = {}
        t1 = time.perf_counter()
        pol_iter = PolicyIteration(
            transitions=probabilities,
            reward=rewards,
            max_iter=10_000,
            eval_type=1,
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

    converge_list = []
    for i in range(1, size):
        poly = PolicyIteration(
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


def plot_convergence(converge_lst):
    x_vals = []
    y_vals = []
    for i, val in enumerate(converge_lst):
        x_vals.append(i)
        y_vals.append(val)
    plt.plot(x_vals, y_vals, label="Max V")
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.title(f"Convergence (PI) Forest")
    plt.grid(visible=True)
    plt.show()



def main():
    # Set basic values
    map_size = 25
    r1 = 4
    r2 = 2
    start_prob = 0.1
    # gamma_range = [.0001, .001, .01, .1, .3, .6, .8, .9, .95, .999]  # TODO: change this
    gamma_range = [i / 10 for i in range(1, 10)]

    probabilities, rewards = instantiate_prob_rewards(
        map_size=map_size,
        r1=r1,
        r2=r2,
        start_prob=start_prob)

    all_iters, converge_list = policy_iteration(probabilities, rewards, gamma_range, map_size)

    converge_lst = []
    for i in range(1, map_size):
        policy_Q = PolicyIteration(transitions=probabilities,
                                   reward=rewards,
                                   max_iter=i,
                                   gamma=0.999,
                                   )
        converge_lst.append(pd.DataFrame(policy_Q.run())['Mean V'].iloc[-1])

    # plot_results(x_values=gamma_range, y_values=[i["iterations"] for i in all_iters], ylabel="Iterations",
    #              title="Gamma & Iteration Values (PI) Forest")
    # plot_results(x_values=gamma_range, y_values=[i["time"] for i in all_iters], ylabel="Time (Seconds)",
    #              title="Gamma & Wall Clock Times (PI) Forest")
    plot_results(x_values=gamma_range, y_values=[i["score"] for i in all_iters], ylabel="Score (Avg)",
                 title="Gamma & Score (PI) Forest")
    plot_convergence(converge_lst=converge_lst)


if __name__ == "__main__":
    main()
