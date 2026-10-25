from screens.base import BaseScreen
from screens.placeholder import (
    AggregationScreen,
    ParameterSelectionScreen,
    SavingScreen,
    TrainingScreen,
)
from screens.preprocessing import PreprocessingScreen


SCREENS: list[type[BaseScreen]] = [
    PreprocessingScreen,
    ParameterSelectionScreen,
    TrainingScreen,
    AggregationScreen,
    SavingScreen,
]

__all__ = ['BaseScreen', 'SCREENS']
