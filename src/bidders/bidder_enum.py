from enum import auto
from src.utils.enums import StrEnum
from src.bidders.bidders import (
    EmpiricalShadedBidder,
    TruthfulBidder,
    ValueLearningBidder,
    PolicyLearningBidder,
    DoublyRobustBidder,
)


class BiddersEnum(StrEnum):
    EMPIRICAL_SHADED_BIDDER = auto()
    TRUTHFUL_BIDDER = auto()
    VALUE_LEARNING_BIDDER = auto()
    POLICY_LEARNING_BIDDER = auto()
    DOUBLY_ROBUST_BIDDER = auto()


BIDDERS_MAPPING = {
    BiddersEnum.EMPIRICAL_SHADED_BIDDER: EmpiricalShadedBidder,
    BiddersEnum.TRUTHFUL_BIDDER: TruthfulBidder,
    BiddersEnum.VALUE_LEARNING_BIDDER: ValueLearningBidder,
    BiddersEnum.POLICY_LEARNING_BIDDER: PolicyLearningBidder,
    BiddersEnum.DOUBLY_ROBUST_BIDDER: DoublyRobustBidder,
}


def get_bidder(bidder_enum):
    return BIDDERS_MAPPING[bidder_enum]
