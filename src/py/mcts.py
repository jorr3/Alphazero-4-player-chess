import numpy as np
import torch

from alphazero_cpp import Node

class MCTS:
    def __init__(self, gameType, neural_net, args):
        self.gameType = gameType
        self.args = args
        self.neural_net = neural_net

    @torch.no_grad()
    def search(self, states, player):
        encoded_states = self.gameType.GetEncodedStates(states, str(self.neural_net.device))
        flat_root_policy = self.get_root_policy(encoded_states)
        root_policy = self.gameType.ParseActionspace(flat_root_policy, states[0].GetTurn())
        self.initialize_games(states, root_policy, player)

        for _ in range(self.args['num_searches']):
            self.process_search_iteration(states)

    def get_root_policy(self, encoded_states):
        root_policy, _ = self.neural_net(encoded_states)
        root_policy = torch.softmax(root_policy, dim=1)
        return self.add_dirichlet_noise(root_policy, self.neural_net.device)

    def add_dirichlet_noise(self, policy, device):
        dirichlet_alpha = self.args['dirichlet_alpha']
        dirichlet_epsilon = self.args['dirichlet_epsilon']
        noise = torch.tensor(
            np.random.dirichlet([dirichlet_alpha] * self.gameType.action_space_size, size=policy.shape[0]),
            device=device, dtype=torch.float32)
        return (1 - dirichlet_epsilon) * policy + dirichlet_epsilon * noise

    def initialize_games(self, states, root_policy, player):
        legal_actions_masks = self.gameType.GetLegalMovesMask(states, str(self.neural_net.device))
        game_policies = root_policy * legal_actions_masks
        num_dims = tuple(range(1, len(self.gameType.state_space_dims) + 1))
        game_policies /= torch.sum(game_policies, dim=num_dims, keepdim=True)

        for i, game_state in enumerate(states):
            game_state.SetRoot(Node(self.args, game_state, player, visit_count=1))
            game_state.GetRoot().Expand(game_policies[i])

    def process_search_iteration(self, games):
        non_terminal_games = self.execute_search_steps(games)
        if non_terminal_games:
            self.update_with_neural_net_predictions(non_terminal_games)

    def execute_search_steps(self, games):
        """Steps through the games until a terminal state is reached. Returns the non-terminal games."""
        return [game for game in games if self.step_through_game(game)]

    def step_through_game(self, game):
        node = game.GetRoot()
        while node.IsFullyExpanded():
            node = node.SelectChild()
        is_terminal, terminal_value = node.GetState().GetTerminated()
        if is_terminal:
            node.Backpropagate(terminal_value)
            return False
        else:
            game.SetNode(node)
            return True

    def update_with_neural_net_predictions(self, games):
        states = [game.GetNode().GetState() for game in games]
        encoded_states = self.gameType.GetEncodedStates(states, str(self.neural_net.device))
        flat_policy, value = self.neural_net(encoded_states)
        flat_policy = torch.softmax(flat_policy, dim=1)

        policy = self.gameType.ParseActionspace(flat_policy, states[0].GetTurn())

        valid_moves_mask = self.gameType.GetLegalMovesMask(states, str(self.neural_net.device))
        policy *= valid_moves_mask
        num_dims = tuple(range(1, len(self.gameType.state_space_dims) + 1))
        policy = policy / torch.sum(policy, dim=num_dims, keepdim=True)

        for i, game in enumerate(games):
            node = game.GetNode()
            node.Expand(policy[i])
            node.Backpropagate(value[i])
