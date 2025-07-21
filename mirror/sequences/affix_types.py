from ..graphs.align_types import AbstractAlignment

class Affix:
    """A candidate affix that has not yet been oriented. 
    
    The affix string can be accessed via methods `call` and `reverse_call`."""

    @staticmethod
    def _call_weight_sequence(sequence: Iterator[tuple[str, str]], placeholders: list[str]) -> Iterator[str]:
        for (x, y) in sequence:
            x_pass = (x in placeholders)
            y_pass = (y in placeholders)
            if x_pass and not(y_pass):
                yield y
            elif not(x_pass) and y_pass:
                yield x
            elif x == y:
                yield x
            else:
                yield f"{x}/{y}"
    
    @classmethod
    def from_alignment(cls, alignment: AbstractAlignment, placeholders: list[str]):
        return cls(
            called_sequence = list(cls._call_affix_from_weight_sequence(alignment.weights(), placeholders)),
            score = alignment.score)

    def __init__(self, called_sequence: list[str], score: float):
        self._called_sequence = called_sequence 
        self._reverse_called_sequence = called_sequence[::-1]
        self._score = score
    
    def call(self) -> str:
        return ' '.join(self._called_sequence)

    def reverse_call(self) -> str:
        return ' '.join(self._reverse_called_sequence)
    
    def score(self) -> float:
        return self._score