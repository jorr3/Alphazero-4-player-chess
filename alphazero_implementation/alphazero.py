import random
import wandb
import torch
import torch.nn.functional as F

from alphazero_implementation.IGame import IGame
from alphazero_implementation.mcts import MCTS
from alphazero_implementation.move import Move
from alphazero_implementation.net import ResNet
from alphazero_implementation.fourpchess import FourPlayerChess
from typing import Type
from tqdm import trange
from chessenv import Player, PlayerColor


class AlphaZero:
    def __init__(self, model, optimizer, gameType: Type[IGame], args):
        self.model = model
        self.optimizer = optimizer
        self.gameType = gameType
        self.args = args
        self.mcts = MCTS(gameType, model, args)

    def selfPlay(self):
        return_memory = []
        game_length = 0
        player = Player(PlayerColor.RED)
        games = [self.gameType() for _ in range(self.args['num_parallel_games'])]

        while len(games) > 0 and game_length < self.args['max_game_length']:
            # print(games[0].state)
            # print("\n")
            states = [g.state for g in games]

            self.mcts.search(states, games, player)

            for i in range(len(games))[::-1]:
                game = games[i]

                action_probs = torch.zeros(self.gameType.action_space_size, device=self.model.device)
                for child in game.root.children:
                    action_probs[child.action_taken.move_index] = child.visit_count
                action_probs /= action_probs.sum()

                game.memory.append((game.root.state, action_probs, player))

                temperature_action_probs = torch.pow(action_probs, 1 / self.args['temperature'])
                temperature_action_probs /= temperature_action_probs.sum()
                action = torch.multinomial(temperature_action_probs, 1).item()

                game.state = self.gameType.take_action(game.state, Move(action, self.gameType.board_size), player)

                is_terminal, terminal_value = self.gameType.get_terminated(game.state)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in game.memory:
                        hist_outcome = terminal_value if hist_player == player else self.gameType.get_opponent_value(
                            terminal_value)
                        return_memory.append((
                            self.gameType.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del games[i]

            game_length += 1
            player = self.gameType.get_opponent(player)

        return return_memory

    import random
    import torch
    import torch.nn.functional as F

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory), batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            # Convert lists of tensors into a single tensor for each category
            state = torch.stack(state).to(dtype=torch.float32, device=self.model.device)
            policy_targets = torch.stack(policy_targets).to(dtype=torch.float32, device=self.model.device)
            value_targets = torch.stack(value_targets).to(dtype=torch.float32, device=self.model.device)
            value_targets = value_targets.view(-1, 1)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value.squeeze(), value_targets.squeeze())
            loss = policy_loss + value_loss

            # Log losses and total loss to wandb
            wandb.log({"Policy Loss": policy_loss.item(),
                       "Value Loss": value_loss.item(),
                       "Total Loss": loss.item()})

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            # torch.save(self.model.state_dict(), f"model_{iteration}_{self.gameType}.pt")
            # torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.gameType}.pt")

if __name__ == "__main__":


    # wandb.init(project="chess_training", name="AlphaZero Chess")
    # time.sleep(5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gameType = FourPlayerChess

    model = ResNet(gameType, 20, 256, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        'max_game_length': 40,
        'C': 2,
        'num_searches': 100,
        'num_iterations': 100,
        'num_selfPlay_iterations': 1,
        'num_parallel_games': 1,
        'num_epochs': 5,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    alphaZero = AlphaZero(model, optimizer, gameType, args)
    alphaZero.learn()