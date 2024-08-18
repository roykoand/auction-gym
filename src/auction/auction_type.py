from abc import ABC, abstractmethod

import numpy as np

from src.utils.enums import StrEnum
from enum import auto


class AllocationMechanism(ABC):
    """Base class for allocation mechanisms"""

    def __init__(self):
        pass

    @abstractmethod
    def allocate(self, bids, num_slots):
        pass


class FirstPrice(AllocationMechanism):
    """(Generalized) First-Price Allocation
    TODO: this works as long as max_slots = 1
        no-support of multiple ad slots"""

    def allocate(self, bids, num_slots):
        winners = np.argsort(-bids)[:num_slots]
        sorted_bids = np.sort(bids)[::-1]
        prices = sorted_bids[:num_slots]
        second_prices = sorted_bids[1 : num_slots + 1]
        return winners, prices, second_prices


class SecondPrice(AllocationMechanism):
    """(Generalized) Second-Price Allocation
    TODO: this works as long as max_slots = 1
        no-support of multiple ad slots"""

    def allocate(self, bids, num_slots):
        winners = np.argsort(-bids)[:num_slots]
        prices = np.sort(bids)[::-1][1 : num_slots + 1]
        return winners, prices, prices


class AuctionType(StrEnum):
    FIRST_PRICE = auto()
    SECOND_PRICE = auto()


AUCTION_TYPE_MAPPING = {
    AuctionType.FIRST_PRICE: FirstPrice(),
    AuctionType.SECOND_PRICE: SecondPrice(),
}


def get_auction(auction_type):
    return AUCTION_TYPE_MAPPING[auction_type]
