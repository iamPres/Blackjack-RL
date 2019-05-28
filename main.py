import numpy as np
import copy

import blackjack
import BJRL

def main():
    print("\nWelcome to my Blackjack RL bot\n")

    while True:
        user = input("1. Game demo\n2. Train\n3. Run agent\n4. Evaluate agent\n5. Quit\n")

        if (user == "1"):
            game_demo()
        elif (user == "2"):
            train_agent()
        elif (user == "3"):
            run_agent()
        elif (user == "4"):
            evaluate_agent()
        elif (user == "5"):
            break
        else:
            print("\n[-] Error: please enter a valid option\n")


def game_demo():
    b = blackjack.blackjack(verbose=True)

    while True:        
        b.deal()

        while True:
            reward, over = b.choose_action(int(input("choose action: ")))

            if over:
                print("[*] Reward:", reward)
                break
                
        if input("\nPress enter to continue\n")!="":
            break

def train_agent():
    agent = BJRL.BJRL()
    agent.train(700000)

    user = input("\nWould you like to save? y/n\n")

    if user == "y":
        agent.save()

def run_agent():
    b = blackjack.blackjack()

    print("[*] Loading agent")
    agent = Agent()
    agent.load_agent()

    while True:        
        b.deal()
        b2 = copy.deepcopy(b)

        print("\n[!] This doesnt really work\n")
        print("\n---Deal---")
        b.print_state()

        over, over2 = False, False

        while True:   
            state1 = b.get_state()
            state2 = b2.get_state()

            print_states(state1, state2)

            if not over:
                reward, over = b.choose_action(int(input("choose action: ")))

            if not over2:
                reward2, over2 = b2.choose_action(agent.get_action(state2))

            if over and over2:

                print_states(state1, state2)

                if (reward > reward2):
                    print("You win!")
                elif (reward < reward2):
                    print("AI wins!")
                else:
                    print("Tie")

                break
                
        if input("\nPress enter to continue\n")!="":
            break

def print_states(state1, state2):
    print("\n       --------Action---------")
    print("          P1     |      AI")

    if len(str(state1[1]))==1:
        print("Dealer:", state1[1] , "       |   ", state2[1])
    else:
        print("Dealer:", state1[1] , "      |   ", state2[1])

    if len(str(state1[0]))==1:
        print("Player:", state1[0], "       |   ", state2[0])
    else:
        print("Player:", state1[0], "      |   ", state2[0])

def evaluate_agent(num_iters=50000):
    b = blackjack.blackjack()

    agent = Agent()    
    agent.load_agent(input("\nEnter filename: ") + ".npy")

    record = np.empty(num_iters+1)

    print("[*] Running samples")

    for i in range(num_iters+1):
        record[i] = agent.run_episode(b)

    wins = np.sum(record == 1)
    ties = np.sum(record == 0)
    losses = np.sum(record == -1)

    print("\n-------| Agent |-------")
    print("After {} iterations:".format(num_iters))
    print("Wins: {}".format(wins))
    print("Ties: {}".format(ties))
    print("Losses: {}".format(losses))
    print("Win probability: {}% | Average player: 44-48%\n".format(round((wins/num_iters) * 100, 2)))
    

class Agent():

    def __init__(self):
        self.policy = np.empty((22, 22, 2))

    def load_agent(self, filename="default.npy"):
        print("\n[*] Loading agent")
        self.policy = np.load(filename)

    def get_action(self, state):
        return np.argmax(self.policy[state[0], state[1]])
    
    def run_episode(self, b):
        b.deal()

        while True:
            agent_sum, dealer_sum = b.get_state()

            #actions array substituted
            action = np.random.choice([0, 1], p=self.policy[agent_sum, dealer_sum])

            reward, over = b.choose_action(action)

            if over:
                break

        return reward

main()