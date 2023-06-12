#!/usr/bin/env python
import argparse
import random

class suit:
    def __init__(self):
        self.cards =[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            10,
            10,
            10,
        ]

    def get_card(self):
        random.shuffle(self.cards)
        return self.cards.pop()

    def get_card_count(self):
        return len(self.cards)
    
class Deck:
    def __init__(self):
        self.cards = [
            suit(), # Hearts
            suit(), # Clubs
            suit(), # Diamonds
            suit()  # Spades
        ]
    
    def get_card(self):
        new_card = None
        random.shuffle(self.cards)
        if self.cards and self.cards[0]:
            new_card = self.cards[0].get_card()
            if len(self.cards[0].cards) == 0:
                del self.cards[0]
        return new_card

    def get_card_count(self):
        card_count = 0
        for suit in self.cards:
            card_count += suit.get_card_count()
        return card_count

class Dealer:
    def __init__(self) -> None:
        self.cards = []

    def give_card(self, card):
        self.cards.append({'value': card, 'seen': False})

    def get_card_sum(self, seeDealerHand=False):
        card_sum = 0 
        if not seeDealerHand:
            # Game is not done so we only know the value of the first dealer card
            card_sum = self.cards[0]['value']
            if card_sum == 1:
                # Ace worth 11 to startf
                card_sum += 10 
            return card_sum
        ace_count = 0
        for card in self.cards:
            current_card_value = card['value']
            if current_card_value == 1:
                current_card_value += 10
                ace_count += 1
            card_sum += current_card_value
            if card_sum > 21 and ace_count:
                # Use the ace as value of 1 instead of 11
                card_sum -= 10
                ace_count -= 1
        return card_sum
    
    def dealer_play(self):
        if self.get_card_sum(True) >= 17:
            return 0 # Stay
        return 1 # Hit
    
    def new_seen_cards(self, seeDealerHand):
        new_cards = []

        # if the game is not done only show 1 dealer card
        if not seeDealerHand:
            if not self.cards[0]['seen']:
                new_cards.append(self.cards[0]['value'])
                self.cards[0]['seen'] = True
            return new_cards
        
        for i in range(len(self.cards)):
            if self.cards[i]['seen'] == False:
                new_cards.append(self.cards[i]['value'])
                self.cards[i]['seen'] = True
        return new_cards

class Player:
    def __init__(self) -> None:
        self.cards = []

    def give_card(self, card):
        self.cards.append({'value': card, 'seen': False})

    def get_card_sum(self):
        card_sum = 0 
        ace_count = 0
        for card in self.cards:
            current_card = card['value']
            if current_card == 1:
                ace_count += 1
                current_card += 10 # ace worth 11
            card_sum += current_card
            if card_sum > 21 and ace_count:
                card_sum -= 10
                ace_count -= 1

        hasAce = False
        if ace_count:
            hasAce = True
        return card_sum, hasAce
    
    def new_seen_cards(self):
        new_cards = []
        for i in range(len(self.cards)):
            if self.cards[i]['seen'] == False:
                new_cards.append(self.cards[i]['value'])
                self.cards[i]['seen'] = True
        return new_cards

class BlackJack:
    def __init__(self, players=1, cardState=False):
        self.deck = Deck()
        self.players = []
        for i in range(players):
            self.players.append(Player())
        self.dealer = Dealer()
        self.cardState = cardState
        self.seen_cards = {
            '1':0,
            '2':0,
            '3':0,
            '4':0,
            '5':0,
            '6':0,
            '7':0,
            '8':0,
            '9':0,
            '10':0,
        }

    def deal_new_hand(self):
        for i in range(len(self.players)):
            self.players[i].cards = []
            self.players[i].give_card(self.deck.get_card())
            self.players[i].give_card(self.deck.get_card())
        self.dealer.cards = []
        self.dealer.give_card(self.deck.get_card())
        self.dealer.give_card(self.deck.get_card())

    def get_seen_cards(self, seeDealerHand=False):
        # Update the cards that have been seen on this hand
        new_cards = []
        for i in range(len(self.players)):
            new_cards += self.players[i].new_seen_cards()
        new_cards += self.dealer.new_seen_cards(seeDealerHand)

        for card in new_cards:
            self.seen_cards[str(card)] += 1
        return self.seen_cards

    def get_game_state(self, isDone=False):
        player_sum, hasAce = self.players[0].get_card_sum()
        player_bust = False
        if player_sum > 21:
            isDone = True
            player_bust = True
        if player_bust:
            # If the player busts we don't get to see the dealers hand
            dealer_sum = self.dealer.get_card_sum(False)
            seen_cards = self.get_seen_cards(False)
        else:
            dealer_sum = self.dealer.get_card_sum(isDone)
            seen_cards = self.get_seen_cards(isDone)
        
        state = [
                player_sum, 
                dealer_sum,
                hasAce
            ]
        
        if self.cardState:
            # add the card counts to the state
            for card in seen_cards:
                state.append(seen_cards[card])

        return state, isDone
    
    def get_reward(self, player_sum, dealer_sum, isDone):
        reward = 0.0
        if player_sum > 21:
            reward = -1.0
        elif dealer_sum > 21:
            reward = 1.0
        elif isDone and player_sum > dealer_sum:
            reward = 1.0
        elif isDone and dealer_sum > player_sum:
            reward = -1.0

        return reward
    
    def reset(self):
        MAX_CARDS_ONE_GAME = 17
        if self.deck.get_card_count() <= MAX_CARDS_ONE_GAME:
            # Shuffle the deck
            self.deck = Deck()
                    # Forget all our seen cards since we are getting a new deck
            self.seen_cards = {
                '1':0,
                '2':0,
                '3':0,
                '4':0,
                '5':0,
                '6':0,
                '7':0,
                '8':0,
                '9':0,
                '10':0,
            }

        self.deal_new_hand()
        state, _ = self.get_game_state()
        return state, None

    def step(self, action):
        isDone = False
        try:
            action = int(action)
            if action == 0:
                isDone = True
                while self.dealer.dealer_play():
                    self.dealer.give_card(self.deck.get_card())
            else:
                self.players[0].give_card(self.deck.get_card())
        except TypeError as error:
            print(f"invalid action: {action} errror {error}")
        state, isDone = self.get_game_state(isDone)
        player_sum = state[0]
        dealer_sum = state[1]
        reward = self.get_reward(player_sum, dealer_sum, isDone)
        return state, reward, isDone, None, None


def input_args():
    parser = argparse.ArgumentParser(
        prog='black_jack_learn',
        description='train a model to learn how to play blackjack'
    )
    parser.add_argument(
        '--cardState',
        action='store_true',
        help="If you want to include seen cards to the game state"
    )
    args = parser.parse_args()

    return args

def print_all_cards():
    while(True):
        i += 1
        card = black_jack.deck.get_card()
        if card:
            print(f"{i}: {card}")
        else:
            break

if __name__ == '__main__':
    args = input_args()
    black_jack = BlackJack(cardState=args.cardState)
    i = 0 

    user_input = None
    games_played = 0
    games_won = 0
    while not user_input == 'q':
        games_played += 1
        print("New hand dealt")
        state, _ = black_jack.reset()
        done = False
        while not done:
            print(f"user sum: {state[0]} user ace: {state[2]} \ndealer card: {state[1]}\n ")
            user_input = input("0 = stay, 1 = hit, q = quit \n:")
            if user_input == 'q':
                break
            state, reward, done, _1, _2 = black_jack.step(user_input)
            if done:
                print(f"user sum: {state[0]}\ndealer sum: {state[1]}\n ")
                if reward >= 1.0:
                    print(f"Win!")
                    games_won += 1
                else:
                    print(f"loss :(")
                print(f"Won: {games_won} of {games_played} win rate: {(games_won/games_played)*100}%\n")

