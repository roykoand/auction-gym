# using python3.9, no StrEnum

from enum import Enum


class StrEnum(str, Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name.replace("_", " ").title().replace(" ", "")

    def __str__(self):
        return str(self.value)
