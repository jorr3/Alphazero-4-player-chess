import itertools


class StateSerializer:
    @staticmethod
    def serialize(state):
        return list(itertools.chain.from_iterable([[str(piece) for piece in p] for p in state.GetPieces()]))
