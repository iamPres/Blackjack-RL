import numpy as np
import random

def build_deck():
        v=0
        deck = np.empty(52, dtype=int)

        for i in range(9):
            v+=1
            deck[i*4:i*4+4]= v

        deck[36:52] = 10

        return deck


class blackjack():   

    def __init__(self, verbose=False):
        self.deck = np.empty(52, dtype=int)
        self.verbose = verbose


    def deal(self):        
        self.deck = build_deck()
        self.end = 51
        self.dealer = 0
        self.agent = 0

        self.dealer, self.deck = self.draw_card(self.deck, self.dealer)
        self.dealer, self.deck = self.draw_card(self.deck, self.dealer)
        self.agent, self.deck = self.draw_card(self.deck, self.agent)

        if (self.verbose==True):
            print("---------Deal----------")
            self.print_state()

    #0: hit, 1: hold
    def choose_action(self, action):
        deck = self.deck
        dealer = self.dealer
        agent = self.agent
        over = False

        if (action==0):
            agent, deck = self.draw_card(deck, agent)

            if(agent > 21):
                over = True
                reward = -1
            else:
                reward = 0                
        else:
            over = True
            
            while (dealer <= 17):
                dealer, deck = self.draw_card(deck, dealer)

            if (dealer > 21 or agent > dealer):
                reward = 1
            elif (agent == dealer):
                reward = 0
            else:
                reward = -1

        self.deck = deck
        self.dealer = dealer
        self.agent = agent

        if (self.verbose==True):
            print("--------Action---------")
            self.print_state()
        
            
        return reward, over


    def draw_card(self, deck, hand):
        choice = random.randint(1, self.end)
        self.end-=1

        hand += deck[choice]
        deck = np.delete(deck, choice)

        return hand, deck

    def get_state(self):
        return self.agent, self.dealer

    def print_state(self, show_deck=False):
        print("Agent:", self.agent)
        print("Dealer:", self.dealer)
        print("Deck count:", len(self.deck))

        if show_deck:
            print("Deck:", self.deck)


