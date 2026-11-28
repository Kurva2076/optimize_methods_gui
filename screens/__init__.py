from screens.base import BaseScreen
from screens.preprocessing import PreprocessingScreen


SCREENS: list[type[BaseScreen]] = [
    PreprocessingScreen,
]

__all__ = ['BaseScreen', 'SCREENS']
