import random
import wandb
import torch
import torch.nn.functional as F
import sys
# import pygame

from alphazero_cpp import Player, PlayerColor, Move, Board

from src.py.fen_parser import parse_board_args_from_fen
from src.py.mcts import MCTS
from src.py.net import ResNet
from tqdm import trange
from fourpchess_interface import FourPlayerChessInterface

from memory_profiler import profile


class AlphaZero:
    def __init__(self, model, optimizer, gameType, args, game_init_args=None, visualize=False):
        self.model = model
        self.optimizer = optimizer
        self.gameType = gameType
        self.args = args
        self.game_init_args = game_init_args
        self.mcts = MCTS(gameType, model, args)
        self.visualize = visualize
        if self.visualize:
            self.ui = FourPlayerChessInterface()

    @profile
    def play(self):
        return_memory = []
        game_length = 0
        player_color = PlayerColor.RED
        states = [
            self.gameType() if not self.game_init_args else self.gameType(*self.game_init_args)
            for _ in range(self.args['num_parallel_games'])
        ]

        while len(states) > 0 and game_length < self.args['max_game_length']:
            print(game_length)

            if self.visualize:
                self.ui.draw_board_state(states[0].state)

            self.mcts.search(states, player_color)

            for i in range(len(states))[::-1]:
                state = states[i]

                if self.visualize:
                    pygame.event.get()

                action_probs = torch.zeros(self.gameType.action_space_size, device=self.model.device)
                for child in state.GetRoot().GetChildren():
                    action_probs[child.GetMoveMade().GetFlatIndex()] = child.GetVisitCount()
                action_probs /= action_probs.sum()

                state.AppendToMemory((state.GetSimpleState(), action_probs, player_color))

                temperature_adjusted_probs = torch.pow(action_probs, 1 / self.args['temperature'])
                temperature_adjusted_probs /= temperature_adjusted_probs.sum()
                action_index = torch.multinomial(temperature_adjusted_probs, 1).item()
                action = Move(action_index)

                state = state.TakeAction(action)

                is_terminal, terminal_value = state.GetTerminated()

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in state.GetMemory():
                        hist_outcome = terminal_value if hist_player == player_color else self.gameType.GetOpponentValue(
                            terminal_value)
                        return_memory.append((
                            self.gameType.GetEncodedStates(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del states[i]

            game_length += 1
            player_color = self.gameType.GetOpponent(player_color)

        return return_memory


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
            for _ in trange(self.args['num_games'] // self.args['num_parallel_games']):
                memory += self.play()

            self.model.train()
            for _ in trange(self.args['num_epochs']):
                self.train(memory)

            # torch.save(self.model.state_dict(), f"model_{iteration}_{self.gameType}.pt")
            # torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.gameType}.pt")

if __name__ == "__main__":

    # wandb.init(project="chess_training", name="AlphaZero Chess")
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    gameType = Board

    board_init_args = parse_board_args_from_fen(Board.start_fen, Board.board_size)

    model = ResNet(gameType, 20, 256, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        'max_game_length': 25 * 4,
        'C': 2,
        'num_searches': 100,
        'num_iterations': 1,
        'num_games': 10,
        'num_parallel_games': 10,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    alphaZero = AlphaZero(model, optimizer, gameType, args, game_init_args=board_init_args, visualize=False, )
    alphaZero.learn()