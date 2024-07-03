import numpy as np
import torch

from alphazero_cpp import Node, BoardPool
from line_profiler_pycharm import profile

from src.py.four_player_chess_board import FourPlayerChess


class MCTS:
    def __init__(self, gameType, neural_net, args):
        self.gameType = gameType
        self.args = args
        self.neural_net = neural_net
        self.board_pool = BoardPool(int(args["pool_size"]))

    @torch.no_grad()
    def search(self, games, player):
        for g in games:
            g.SetRootNode(Node(self.args["C"], g, player, visit_count=1))

        for _ in range(self.args["num_searches"]):
            non_terminal_games = [g for g in games if self.choose_leaf(g)]
            self.step(non_terminal_games)

    def get_root_policy(self, encoded_states):
        root_policy, _ = self.neural_net(encoded_states)
        root_policy = torch.softmax(root_policy, dim=1)
        return self.add_dirichlet_noise(root_policy, self.neural_net.device)

    def add_dirichlet_noise(self, policy, device):
        dirichlet_alpha = self.args["dirichlet_alpha"]
        dirichlet_epsilon = self.args["dirichlet_epsilon"]
        noise = torch.tensor(
            np.random.dirichlet(
                [dirichlet_alpha] * self.gameType.action_space_size,
                size=policy.shape[0],
            ),
            device=device,
            dtype=torch.float32,
        )
        return (1 - dirichlet_epsilon) * policy + dirichlet_epsilon * noise

    def choose_leaf(self, game):
        node = game.GetRootNode()

        while node.IsExpanded():
            node = node.SelectChild()

        is_terminal, terminal_value = node.GetState().GetTerminated()
        if is_terminal:
            node.Backpropagate(terminal_value)
            return False
        else:
            game.SetNode(node)
            return True

    @profile
    def step(self, games):
        if len(games) == 0:
            return

        states = [game.GetNode().GetState() for game in games]
        encoded_states = self.gameType.GetEncodedStates(states, str(self.neural_net.device))
        flat_policy, value = self.neural_net(encoded_states)
        flat_policy = torch.softmax(flat_policy, dim=1)

        policy = self.gameType.ParseActionspace(flat_policy, states[0].GetTurn())

        valid_moves_mask = FourPlayerChess.get_legal_moves_mask(
            states, str(self.neural_net.device)
        )
        policy *= valid_moves_mask[: len(states)]
        num_dims = tuple(range(1, len(self.gameType.state_space_dims) + 1))
        policy = policy / torch.sum(policy, dim=num_dims, keepdim=True)

        nodes = [game.GetNode() for game in games]
        Node.BackpropagateNodes(nodes, value.squeeze(1))
        self.expand(nodes, policy)

    @profile
    def expand(self, nodes, policy_batch):
        policy_batch_cpu = policy_batch.cpu()
        non_zero_indices_batch = torch.nonzero(policy_batch).tolist()

        non_zero_indices = torch.nonzero(policy_batch_cpu, as_tuple=True)
        non_zero_values = policy_batch_cpu[non_zero_indices].tolist()

        Node.ExpandNodes(nodes, policy_batch_cpu, non_zero_indices_batch, non_zero_values, self.board_pool)
