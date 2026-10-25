from screens.base import BaseScreen
from screens.parameter_selection import ParameterSelectionScreen
from screens.placeholder import AggregationScreen, SavingScreen, TrainingScreen
from screens.preprocessing import PreprocessingScreen


SCREENS: list[type[BaseScreen]] = [
    PreprocessingScreen,
    ParameterSelectionScreen,
    TrainingScreen,
    AggregationScreen,
    SavingScreen,
]

__all__ = ['BaseScreen', 'SCREENS']
