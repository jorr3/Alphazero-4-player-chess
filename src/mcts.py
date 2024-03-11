import numpy as np
import torch

from chessenv import Player, PlayerColor

from node import Node

class MCTS:
    def __init__(self, gameType, neural_net, args):
        self.gameType = gameType
        self.args = args
        self.neural_net = neural_net

    @torch.no_grad()
    def search(self, states, games, player):
        encoded_states = self.gameType.get_encoded_states(states, self.neural_net.device)
        root_policy = self.get_root_policy(encoded_states)
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
        for i, game in enumerate(games):
            legal_actions_mask = self.gameType.get_legal_actions_mask(states[i], self.neural_net.device)
            game_policy = root_policy[i] * legal_actions_mask
            game_policy /= torch.sum(game_policy)
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
        policy, value = self.neural_net(self.gameType.get_encoded_states(states, self.neural_net.device))
        policy = torch.softmax(policy, dim=1)

        for i, game in enumerate(games):
            node = game.node
            game_policy = self.apply_legal_actions_mask(policy[i], node.state, self.neural_net.device)
            node.expand(game_policy)
            node.backpropagate(value[i])

    def apply_legal_actions_mask(self, policy, state, device):
        valid_moves = self.gameType.get_legal_actions_mask(state, device)
        policy *= valid_moves
        return policy / torch.sum(policy)
