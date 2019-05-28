import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import blackjack


class BJRL():

    A = np.empty(2, dtype=int)
    gamma = 0.5
    epsilon = 1
    epsilon_decay = 0.99999
    epsilon_min = 0.01
    alpha = 0.01

    def __init__(self):
        self.A[0], self.A[1] = 0, 1
        self.policy = np.full((22, 22, 2), 0.5)
        #self.N = np.zeros((22, 22, 2))
        self.V = np.zeros((22, 22))
        self.Q = np.zeros((22, 22, 2))
        self.na = len(self.A)

        self.average = []
        self.epsilon_hist = []
        self.reward_hist = []


    def train(self, num_iters=10000):       
        plt.figure(1)
        plt.title("State-Value function")
        self.ax = plt.axes(projection='3d')
        self.ax.view_init(28, -131)
        self.plot()

        plt.figure(2)
        plt.title("Epsilon vs. Time")
        plt.plot(0, self.epsilon)
        input()

        b = blackjack.blackjack()
        t = 0
        
        for it in range(num_iters+1):
            history = self.run_episode(b)
            self.evaluate(history)
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

            if t == 0:
                t=10000

                self.plot()
                av_reward = np.sum(self.average)/t
                print("[*] Iteration:", it)
                print("[*] Average reward:", av_reward)

                self.average=[]
                self.epsilon_hist.append(self.epsilon)
                self.reward_hist.append(av_reward)
               
            t-=1
        
        self.epsilon = 0

        for i in range(len(self.Q)):
            for i2 in range(len(self.Q[0])):
                self.improve([i, i2])

    
    def run_episode(self, b):
        history = []

        b.deal()

        while True:
            agent_sum, dealer_sum = b.get_state()

            action = np.random.choice(self.A, p=self.policy[agent_sum, dealer_sum])

            reward, over = b.choose_action(action)

            history.append([reward, agent_sum, dealer_sum, action])
            self.average.append(reward)
            #self.N[agent_sum, dealer_sum, action] += 1

            if over:
                break

        return history


    def evaluate(self, history):
        
        for _ in range(len(history)):
            total, index = self.disc_value(history)

            self.Q[index[0], index[1], index[2]] = self.Q[index[0], index[1], index[2]] + self.alpha * (total - self.Q[index[0], index[1], index[2]])
            self.V[index[0], index[1]] = self.V[index[0], index[1]] + self.alpha * (total - self.V[index[0], index[1]])

            self.improve(index)


    def improve(self, index):
        best_action = np.argmax(self.Q[index[0], index[1]])
        self.policy[index[0], index[1]] = np.full(self.na, (self.epsilon/self.na))
        self.policy[index[0], index[1], best_action] = (self.epsilon/self.na) + 1 - self.epsilon
        
    def disc_value(self, history): 

        v, index = np.split(history.pop(0), [1])
        v = float(v)

        for ind, val in enumerate(history):
            v+=(self.gamma ** ind) * val[0]


        return v, index

    def plot(self):
        plt.figure(1)
        plt.cla()
        V = self.V[2:,2:]

        (x, y) = np.meshgrid(np.arange(V.shape[0]), np.arange(V.shape[1]))
        self.ax.plot_wireframe(x, y, V, cmap='binary')

        plt.draw()
        plt.xlabel("Dealer_sum")
        plt.ylabel("Agent_sum")
        plt.pause(0.000001)

        plt.figure(2)
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(self.epsilon_hist)), self.epsilon_hist)
        plt.ylabel("Epsilon")

        plt.subplot(2, 1 , 2)
        plt.plot(np.arange(len(self.reward_hist)), self.reward_hist)
        plt.xlabel("Iterations")
        plt.ylabel("Average Reward")
        plt.pause(0.000001)

    def save(self, filename="default"):
        np.save(filename, self.policy)
        print("\n[*] Model save successfully\n")

