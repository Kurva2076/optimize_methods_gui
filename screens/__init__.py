from screens.base import BaseScreen
from screens.parameter_selection import ParameterSelectionScreen
from screens.placeholder import AggregationScreen, SavingScreen
from screens.preprocessing import PreprocessingScreen
from screens.training import TrainingScreen


SCREENS: list[type[BaseScreen]] = [
    PreprocessingScreen,
    ParameterSelectionScreen,
    TrainingScreen,
    AggregationScreen,
    SavingScreen,
]

__all__ = ['BaseScreen', 'SCREENS']
