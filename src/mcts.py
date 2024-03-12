import numpy as np
import torch


from node import Node

class MCTS:
    def __init__(self, gameType, neural_net, args):
        self.gameType = gameType
        self.args = args
        self.neural_net = neural_net

    @torch.no_grad()
    def search(self, states, games, player):
        encoded_states = self.gameType.get_encoded_states(states, self.neural_net.device)
        flat_root_policy = self.get_root_policy(encoded_states)
        root_policy = self.gameType.parse_actionspace(flat_root_policy, states[0].GetTurn(), self.neural_net.device)
        self.initialize_games(games, states, root_policy, player)

        for _ in range(self.args['num_searches']):
            self.process_search_iteration(games)

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

    def initialize_games(self, games, states, root_policy, player):
        legal_actions_masks = self.gameType.get_legal_actions_mask(states, self.neural_net.device)
        game_policies = root_policy * legal_actions_masks
        num_dims = tuple(range(1, len(self.gameType.state_space_dims) + 1))
        game_policies /= torch.sum(game_policies, dim=num_dims, keepdim=True)

        for i, game in enumerate(games):
            game_policy = game_policies[i]
            game.root = Node(self.gameType, self.args, states[i], player, visit_count=1)
            game.root.expand(game_policy)

    def process_search_iteration(self, games):
        non_terminal_games = self.execute_search_steps(games)
        if non_terminal_games:
            self.update_with_neural_net_predictions(non_terminal_games)

    def execute_search_steps(self, games):
        """Steps through the games until a terminal state is reached. Returns the non-terminal games."""
        return [game for game in games if self.step_through_game(game)]

    def step_through_game(self, game):
        node = game.root
        while node.is_fully_expanded():
            node = node.select_child()
        is_terminal, terminal_value = self.gameType.get_terminated(node.state, node.turn)
        if is_terminal:
            node.backpropagate(terminal_value)
            return False
        else:
            game.node = node
            return True

    def update_with_neural_net_predictions(self, games):
        states = [game.node.state for game in games]
        encoded_states = self.gameType.get_encoded_states(states, self.neural_net.device)
        flat_policy, value = self.neural_net(encoded_states)
        flat_policy = torch.softmax(flat_policy, dim=1)

        policy = self.gameType.parse_actionspace(flat_policy, states[0].GetTurn(), self.neural_net.device)

        valid_moves_mask = self.gameType.get_legal_actions_mask(states, self.neural_net.device)
        policy *= valid_moves_mask
        num_dims = tuple(range(1, len(self.gameType.state_space_dims) + 1))
        policy = policy / torch.sum(policy, dim=num_dims, keepdim=True)

        non_zero_indices = torch.nonzero(policy, as_tuple=True)
        probs = policy[non_zero_indices]

        for i, game in enumerate(games):
            node = game.node
            node.expand(policy[i])
            node.backpropagate(value[i])

