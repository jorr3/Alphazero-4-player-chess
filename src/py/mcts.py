import numpy as np
import torch

from alphazero_cpp import Node, BoardPool, GameResult
from line_profiler_pycharm import profile


class MCTS:
    def __init__(self, gameType, neural_net, args):
        self.gameType = gameType
        self.args = args
        self.neural_net = neural_net
        self.board_pool = BoardPool(int(args["pool_size"]))

    @torch.no_grad()
    @profile
    def search(self, games):
        def get_expandable_leaves(non_terminal_nodes):
            expandable_leaves = []
            for node in non_terminal_nodes[:]:
                leaf = node.ChooseLeaf()
                if leaf is None:
                    non_terminal_nodes.remove(node)
                else:
                    expandable_leaves.append(leaf)
            return expandable_leaves

        root_nodes = []
        for game in games:
            root_node = Node(self.args["C"], game, visit_count=1)
            game.SetRootNode(root_node)
            root_nodes.append(root_node)

        non_terminal_root_nodes = root_nodes[:]

        for _ in range(self.args["num_searches"]):
            expandable_leaves = get_expandable_leaves(non_terminal_root_nodes)
            self.step(expandable_leaves)

        for l in root_nodes:
            assert len(l.GetChildren()) > 0

        return root_nodes

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

    @profile
    def step(self, leaves):
        if len(leaves) == 0:
            return

        states = [leave.GetState() for leave in leaves]

        encoded_states = self.gameType.GetEncodedStates(states, str(self.neural_net.device))
        flat_policy, value = self.neural_net(encoded_states)
        flat_policy = torch.softmax(flat_policy, dim=1)

        policy = self.gameType.ParseActionspace(flat_policy, states[0].GetTurn())

        valid_moves_mask = self.gameType.get_legal_moves_mask(
            states, str(self.neural_net.device)
        )
        policy *= valid_moves_mask#[: len(states)]
        num_dims = tuple(range(1, len(self.gameType.state_space_dims) + 1))
        policy = policy / torch.sum(policy, dim=num_dims, keepdim=True)

        Node.BackpropagateNodes(leaves, value.squeeze(1))
        self.expand(leaves, policy)

    @profile
    def expand(self, leaves, policy_batch):
        policy_batch_cpu = policy_batch.cpu()
        non_zero_indices_batch = torch.nonzero(policy_batch).tolist()

        non_zero_indices = torch.nonzero(policy_batch_cpu, as_tuple=True)
        non_zero_values = policy_batch_cpu[non_zero_indices].tolist()

        Node.ExpandNodes(leaves, policy_batch_cpu, non_zero_indices_batch, non_zero_values, self.board_pool)
