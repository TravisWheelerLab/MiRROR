
from .fragments.types import TargetMassStateSpace
from .alignment import AlignmentResult

class EnumerationResult:
    pass

class EnumerationParams:

    @classmethod
    def from_config(cls, *args, **kwargs):
        pass

def enumerate_candidates(
    algn: AlignmentResult,
    targets: TargetMassStateSpace,
    params: EnumerationParams,
    verbose: bool = False,
) -> EnumerationResult:
    pass
