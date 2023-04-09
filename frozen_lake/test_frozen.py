from jons_hiive.hiive import mdptoolbox
import numpy as np
import matplotlib.pyplot as plt
from jons_hiive.hiive.mdptoolbox.example import openai
from gymnasium.envs.toy_text import frozen_lake


def PI_VI_plot(title, gamma_arr, time_array, list_scores, iters):
    plt.figure()
    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title(title + ' - Running Time vs.Gamma')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title(title + ' - Average Reward vs.Gamma')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iterations to Converge')
    plt.title(title + ' - Iterations to Converge vs.Gamma')
    plt.grid()
    plt.show()


def policy_iteration():
    size = 20
    num_iter = 10

    r1 = 4
    r2 = 2
    p = 0.1
    frozen_map = frozen_lake.generate_random_map(size=size, p=0.9, seed=909)
    P, R = openai(
        env_name="FrozenLake-v1",
        desc=frozen_map
    )
    values_ls = [0] * num_iter
    policy = [0] * num_iter
    iters = [0] * num_iter
    runing_time_ls = [0] * num_iter
    gamma_ls = [0] * num_iter

    for i in range(1, num_iter):
        # gamma = (i + 0.5) / num_iter
        gamma = (i) / 10
        pi = mdptoolbox.mdp.ValueIteration(P, R, gamma)
        pi.run()
        gamma_ls[i] = gamma
        values_ls[i] = np.sum(pi.V)
        policy[i] = pi.policy
        iters[i] = pi.iter
        runing_time_ls[i] = pi.time

        # print('gamma=',gamma, 'policy:', pi.policy)
        # print('gamma=',gamma, 'value:', pi.V)

    PI_VI_plot('Frozen (Value Iteration)', gamma_ls, runing_time_ls, values_ls, iters)


def value_iteration():
    size = 20
    num_iter = 5
    r1 = 4
    r2 = 2
    p = 0.1

    frozen_map = frozen_lake.generate_random_map(size=size, p=0.9, seed=909)
    P, R = openai(
        env_name="FrozenLake-v1",
        desc=frozen_map)
    values_ls = [0] * num_iter
    policy = [0] * num_iter
    iters = [0] * num_iter
    runing_time_ls = [0] * num_iter
    gamma_ls = [0] * num_iter

    for i in range(0, num_iter):
        gamma = (i + 0.5) / num_iter
        pi = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
        pi.run()
        gamma_ls[i] = gamma
        values_ls[i] = np.sum(pi.V)
        policy[i] = pi.policy
        iters[i] = pi.iter
        runing_time_ls[i] = pi.time

    PI_VI_plot('Frozen (Policy Iteration)', gamma_ls, runing_time_ls, values_ls, iters)


if __name__ == "__main__":
    policy_iteration()
    value_iteration()
