from enum import auto
from src.utils.enums import StrEnum
from src.ad_allocation.ad_allocators import (
    PytorchLogisticRegressionAllocator,
    OracleAllocator,
)


class AdAllocationEnum(StrEnum):
    ORACLE_ALLOCATOR = auto()
    PYTORCH_LOGISTIC_REGRESSION_ALLOCATOR = auto()

AD_ALLOCATION_MAPPING = {
    AdAllocationEnum.ORACLE_ALLOCATOR: OracleAllocator,
    AdAllocationEnum.PYTORCH_LOGISTIC_REGRESSION_ALLOCATOR: PytorchLogisticRegressionAllocator,
}

def get_ad_allocator(allocator_enum):
    return AD_ALLOCATION_MAPPING[allocator_enum]
