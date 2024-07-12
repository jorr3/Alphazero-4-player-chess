import multiprocessing as mp
import time
import torch
import torch.nn.functional as F
from line_profiler_pycharm import profile
from tqdm import trange

from replay_buffer import ReplayBuffer
from alphazero_cpp import Player, PlayerColor, Move, MemoryEntry, GameResult
from src.py.fen_parser import parse_board_args_from_fen
from src.py.mcts import MCTS
from src.py.net import ResNet
from fourpchess_interface import FourPlayerChess, run_ui
import wandb
from serialized_storage import StateSerializer
from torch.cuda.amp import GradScaler, autocast


class AlphaZero:
    def __init__(
        self, model, optimizer, gameType, args, game_init_args=None, visualize=False
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1000, gamma=0.1
        )
        self.gameType = gameType
        self.args = args
        self.game_init_args = game_init_args
        self.mcts = MCTS(gameType, model, args)
        self.visualize = visualize
        self.experience_buffer = ReplayBuffer(self.args["replay_buffer_capacity"])
        self.validation_buffer = ReplayBuffer(self.args["validation_buffer_capacity"])

        if self.visualize:
            self.queue = mp.Queue()
            self.ui_process = mp.Process(target=run_ui, args=(self.queue,))
            self.ui_process.start()

        wandb.init(
            project="alphazero",
            config={
                **self.args,
                "board_size": (gameType.nRows(), gameType.nCols()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "scheduler_step_size": self.scheduler.step_size,
                "scheduler_gamma": self.scheduler.gamma,
            },
        )

    def handle_terminal_state(self, state, rewardGen):
        """
        :param state: terminal state where GetTurn() returns the losing player
        """
        split_ratio = self.args["replay_buffer_capacity"] / (
            self.args["replay_buffer_capacity"]
            + self.args["validation_buffer_capacity"]
        )

        for entry in state.GetMemory():
            reward = rewardGen(entry.state.GetTurn().GetTeam())
            buffer = (
                self.experience_buffer
                if torch.rand(1).item() < split_ratio
                else self.validation_buffer
            )
            buffer.add(
                (
                    state,
                    self.gameType.GetEncodedState(
                        entry.state, str(self.model.device)
                    ).squeeze(0),
                    entry.action,
                    reward,
                )
            )

    @profile
    def play(self):
        self.queue.put("new batch")

        states = [
            (
                self.gameType()
                if not self.game_init_args
                else self.gameType(*self.game_init_args)
            )
            for _ in range(self.args["num_parallel_games"])
        ]

        serialized_games = [[] for _ in range(len(states))]

        for game_length in trange(self.args["max_game_length"]):

            start_time = time.time()

            root_nodes = self.mcts.search(states)

            for i in reversed(range(len(states))):
                state = states[i]

                action_probs = torch.zeros(
                    self.gameType.action_space_size, device=self.model.device
                )

                for child in root_nodes[i].GetChildren():
                    action_probs[child.GetMoveMade().GetFlatIndex()] = child.GetVisitCount()
                action_probs /= action_probs.sum()

                state.AppendToMemory(MemoryEntry(state, action_probs))

                temperature_adjusted_probs = torch.pow(
                    action_probs, 1 / self.args["temperature"]
                )
                temperature_adjusted_probs /= temperature_adjusted_probs.sum()
                action_index = torch.multinomial(temperature_adjusted_probs, 1).item()
                action = Move(action_index)

                next_state = state.TakeAction(action)
                next_state.SetRootState(state.GetRootState())
                game_state = next_state.GetGameResult()

                serialized_state = StateSerializer.serialize(state)
                serialized_games[i].append(serialized_state)

                if game_state != GameResult.IN_PROGRESS:
                    losing_team = state.GetTurn().GetTeam()
                    self.handle_terminal_state(
                        state,
                        lambda team: (
                            1
                            if team != losing_team
                            else self.gameType.GetOpponentValue(1)
                        ),
                    )
                    serialized_state = StateSerializer.serialize(next_state)
                    serialized_games[i].append(serialized_state)
                    self.queue.put(serialized_games[i])
                    del states[i]
                    del serialized_games[i]
                else:
                    states[i] = next_state

            end_time = time.time()
            elapsed_time = end_time - start_time

            games_remaining = len(states)

            if games_remaining > 0:
                avg_time_per_game = elapsed_time / games_remaining
                # print(
                #     f"Average time per game for move {game_length + 1}: {avg_time_per_game:.4f} seconds"
                # )

            if len(states) == 0:
                return

        wandb.log({"Games Left at Termination": len(states)})
        for i in reversed(range(len(states))):
            state = states[i]
            curr_team = state.GetTurn().GetTeam()
            heur_val = (
                    state.CalculateHeuristic(curr_team)
                    * self.args["heuristic_weight"]
            )
            self.handle_terminal_state(
                state,
                lambda team: (
                    heur_val
                    if team == curr_team
                    else self.gameType.GetOpponentValue(heur_val)
                ),
            )
            serialized_state = StateSerializer.serialize(state)
            serialized_games[i].append(serialized_state)
            self.queue.put(serialized_games[i])


    def train(self):
        if len(self.experience_buffer) < self.args["batch_size"]:
            return

        for _ in range(0, len(self.experience_buffer), self.args["batch_size"]):
            sample = self.experience_buffer.sample(self.args["batch_size"])
            _, encoded_state, policy_targets, value_targets = zip(*sample)

            encoded_state = torch.stack(encoded_state).to(
                dtype=torch.float32, device=self.model.device
            )
            policy_targets = torch.stack(policy_targets).to(
                dtype=torch.float32, device=self.model.device
            )
            value_targets = torch.tensor(
                value_targets, dtype=torch.float32, device=self.model.device
            ).view(-1, 1)

            # Forward pass without autocast
            out_policy, out_value = self.model(encoded_state)
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value.squeeze(), value_targets.squeeze())
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # Update the learning rate after optimizer step

            total_norm = 0.0
            for p in self.model.parameters():
                param_norm = p.grad.data.norm(2) if p.grad is not None else 0.0
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            wandb.log(
                {
                    "Learning Rate": self.optimizer.param_groups[0]["lr"],
                    "Policy Loss": policy_loss.item(),
                    "Value Loss": value_loss.item(),
                    "Total Loss": loss.item(),
                    "Replay Buffer Size": len(self.experience_buffer),
                    "Gradient Norm": total_norm,
                }
            )

    def validate(self):
        if len(self.validation_buffer) < self.args["batch_size"]:
            return

        with torch.no_grad():
            sample = self.validation_buffer.sample(self.args["batch_size"])
            _, state_t, policy_targets, value_targets = zip(*sample)

            state_t = torch.stack(state_t).to(
                dtype=torch.float32, device=self.model.device
            )
            policy_targets = torch.stack(policy_targets).to(
                dtype=torch.float32, device=self.model.device
            )
            value_targets = torch.tensor(
                value_targets, dtype=torch.float32, device=self.model.device
            ).view(-1, 1)

            # Forward pass without autocast
            out_policy, out_value = self.model(state_t)
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value.squeeze(), value_targets.squeeze())
            loss = policy_loss + value_loss

            wandb.log(
                {
                    "Validation Policy Loss": policy_loss.item(),
                    "Validation Value Loss": value_loss.item(),
                    "Validation Total Loss": loss.item(),
                    "Validation Replay Buffer Size": len(self.validation_buffer),
                }
            )

    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            self.model.eval()
            for _ in range(self.args["num_games"] // self.args["num_parallel_games"]):
                self.play()

            self.model.train()
            self.train()

            self.model.eval()
            self.play()
            self.validate()

            # TODO: log the predicted value

        if self.visualize:
            self.ui_process.terminate()
            self.ui_process.join()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gameType = FourPlayerChess
    board_init_args = parse_board_args_from_fen(
        FourPlayerChess.start_fen, FourPlayerChess.nCols()
    )

    model = ResNet(gameType, 15, 256, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        "pool_size": 10,  # 20e5,
        "max_game_length": 80,
        "C": 3,
        "num_searches": 50,
        "num_iterations": 1000,
        "num_games": 100,
        "num_parallel_games": 100,
        "batch_size": 512,
        "temperature": 1.1,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.2,
        "heuristic_weight": 0.02,
        "replay_buffer_capacity": 150_000,
        "validation_buffer_capacity": 30_000,
    }

    # TODO: make the starting state as simple as possible and if it doesnt learn that there is 100 percent something wrong

    alphaZero = AlphaZero(
        model, optimizer, gameType, args, game_init_args=board_init_args, visualize=True
    )
    alphaZero.learn()
