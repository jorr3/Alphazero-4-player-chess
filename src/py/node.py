import torch
from line_profiler_pycharm import profile

from src.move import Move
from algos import select_child


class Node:
    def __init__(self, game, args, state, turn, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.turn = turn

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    @profile
    def select_child(self):
        children_visit_counts = [child.visit_count for child in self.children]
        children_values = [child.value_sum / child.visit_count if child.visit_count > 0 else 0 for child in self.children]
        children_priors = [child.prior for child in self.children]

        # Call the C++ select_child function
        best_child_index = select_child(
            children_visit_counts,
            children_values,
            children_priors,
            self.visit_count,
            self.args['C']
        )

        return self.children[best_child_index] if best_child_index != -1 else None

    def expand(self, policy):
        non_zero_indices = torch.nonzero(policy, as_tuple=False).tolist()
        # Now non_zero_indices is a list of lists. Each sub-list represents [action_plane, from_row, from_col]

        for idx in non_zero_indices:
            # Since idx is already a list, we can unpack it directly without converting in the loop
            action_plane, from_row, from_col = idx

            move = Move.from_index(action_plane, from_row, from_col, self.turn)

            prob = policy[action_plane, from_row, from_col].item()

            child_turn = self.game.get_opponent(self.turn)
            child_state = self.game.take_action(self.state, move, self.turn)

            child = Node(self.game, self.args, child_state, child_turn, self, move, prob)
            self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)
