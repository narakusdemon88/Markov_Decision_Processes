"""
Hyperparameter tuning for assignment 4
Jon-Erik Akashi (jakashi3@gatech.edu)
"""
from frozen_lake.jons_hiive.hiive.mdptoolbox.example import openai
import numpy as np
from hiive.mdptoolbox.mdp import QLearning
import pandas as pd


def main():
    prob, rew = openai(
        env_name="FrozenLake-v1",
        desc=["SFHFF",
              "FFFFH",
              "FFFFF",
              "FFFFF",
              "FFFFG"]
    )

    gamma_range = [0.0001,
                   0.001,
                   0.01,
                   0.1,
                   0.2,
                   0.3,
                   0.4,
                   0.5,
                   0.6,
                   0.7,
                   0.8,
                   0.9,
                   0.91,
                   0.92,
                   0.93,
                   0.94,
                   0.95,
                   0.999]  # discount factor
    alpha_range = [i / 100 for i in range(1, 100)]  # learning rate
    epsilon_range = [i / 100 for i in range(1, 100)]  # probability of taking a random action
    avg_rewards = []
    df = pd.DataFrame()

    for gamma in gamma_range:
        for alpha in alpha_range:
            for epsilon in epsilon_range:
                ql = QLearning(prob,
                               rew,
                               gamma=gamma,
                               alpha=alpha,
                               epsilon=epsilon,
                               n_iter=10000
                               )
                ql.run()
                avg_reward = np.mean(ql.run_stats[-1]["Reward"])
                avg_rewards.append(avg_reward)
                one_row = {
                    "Gamma": gamma,
                    "Alpha": alpha,
                    "Epsilon": epsilon,
                    "Avg": avg_reward
                }
                df = df.append(one_row, ignore_index=True)
                df.to_csv("results.csv", index=False)
                print(f"gamma={gamma}, alpha={alpha}, epsilon={epsilon}, mean reward={avg_reward}")

    best_i = np.argmax(avg_rewards)
    best_gamma = gamma_range[best_i // (len(alpha_range) * len(epsilon_range))]
    best_alpha = alpha_range[(best_i // len(epsilon_range)) % len(alpha_range)]
    best_epsilon = epsilon_range[best_i % len(epsilon_range)]
    print(f"best gamma: {best_gamma}, best alpha: {best_alpha}, best epsilon: {best_epsilon}")


if __name__ == "__main__":
    main()
