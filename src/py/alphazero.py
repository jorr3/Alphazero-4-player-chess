import random
import wandb
import torch
import torch.nn.functional as F
import pygame
import time
from replay_buffer import ReplayBuffer

from alphazero_cpp import Player, PlayerColor, Move, Board, MemoryEntry
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
        self.experience_buffer = ReplayBuffer(self.args["buffer_capacity"])
        if self.visualize:
            self.ui = FourPlayerChessInterface()

    def handle_terminal_state(self, state, player_color, terminal_value):
        reward = 1 if terminal_value == player_color else -1
        for entry in state.GetMemory():
            val = reward if entry.color == player_color else self.gameType.GetOpponentValue(reward)
            self.experience_buffer.add(
                (
                    self.gameType.GetEncodedStates([entry.simpleState.pieces], player_color, str(self.model.device)),
                    entry.stateTensor,
                    val,
                )
            )

    def play(self):
        game_length = 0
        player_color = PlayerColor.RED
        states = [
            (self.gameType() if not self.game_init_args else self.gameType(*self.game_init_args))
            for _ in range(self.args["num_parallel_games"])
        ]

        while len(states) > 0:
            print(game_length)
            start_time = time.time()

            if self.visualize:
                self.ui.draw_board_state(states[0])

            self.mcts.search(states, player_color)

            for i in reversed(range(len(states))):
                state = states[i]

                if self.visualize:
                    pygame.event.get()

                action_probs = torch.zeros(self.gameType.action_space_size, device=self.model.device)
                for child in state.GetRootNode().GetChildren():
                    action_probs[child.GetMoveMade().GetFlatIndex()] = child.GetVisitCount()
                action_probs /= action_probs.sum()

                state.AppendToMemory(MemoryEntry(state, action_probs, player_color))

                temperature_adjusted_probs = torch.pow(action_probs, 1 / self.args["temperature"])
                temperature_adjusted_probs /= temperature_adjusted_probs.sum()
                action_index = torch.multinomial(temperature_adjusted_probs, 1).item()
                action = Move(action_index)

                next_state = state.TakeAction(action)
                next_state.SetRootState(state.GetRootState())
                is_terminal, terminal_value = next_state.GetTerminated()

                if is_terminal:
                    self.handle_terminal_state(state, player_color, terminal_value)
                    del states[i]
                else:
                    states[i] = next_state

            end_time = time.time()
            elapsed_time = end_time - start_time
            avg_time_per_game = elapsed_time / len(states)
            print(f"Average time per game for move {game_length}: {avg_time_per_game:.4f} seconds")

            game_length += 1
            player_color = self.gameType.GetOpponent(player_color)

    def train(self):
        if len(self.experience_buffer) < self.args["batch_size"]:
            return

        for _ in range(0, len(self.experience_buffer), self.args["batch_size"]):
            sample = self.experience_buffer.sample(self.args["batch_size"])
            state, policy_targets, value_targets = zip(*sample)

            state = torch.stack(state).to(dtype=torch.float32, device=self.model.device)
            policy_targets = torch.stack(policy_targets).to(dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device).view(-1, 1)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value.squeeze(), value_targets.squeeze())
            loss = policy_loss + value_loss

            wandb.log({"Policy Loss": policy_loss.item(), "Value Loss": value_loss.item(), "Total Loss": loss.item()})

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            self.model.eval()
            for _ in range(self.args["num_games"] // self.args["num_parallel_games"]):
                self.play()

            self.model.train()
            for _ in range(self.args["num_epochs"]):
                self.train()

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gameType = Board
    board_init_args = parse_board_args_from_fen(Board.start_fen, Board.board_size)

    model = ResNet(gameType, 20, 256, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        "pool_size": 30e5,
        "max_game_length": 25 * 4,
        "C": 2,
        "num_searches": 100,
        "num_iterations": 15,
        "num_games": 1000,
        "num_parallel_games": 50,
        "num_epochs": 10,
        "batch_size": 128,
        "temperature": 1.1,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3,
        "buffer_capacity": 10000,  # Add buffer capacity for experience buffer
    }

    alphaZero = AlphaZero(model, optimizer, gameType, args, game_init_args=board_init_args, visualize=True)
    alphaZero.learn()

