from Lab7.B.epsilon_greedy_bandit import simulateBandit
from Lab7.B.binaryBanditA import binaryBanditA
from Lab7.B.binaryBanditB import binaryBanditB

if __name__ == "__main__":
    epsilon = 0.1
    episodes = 1000
    rewards, qValues = simulateBandit(epsilon, episodes, binaryBanditA, binaryBanditB)
    print(f"Estimated q-values: {qValues}")
    print(f"Total rewards: {sum(rewards)}") 