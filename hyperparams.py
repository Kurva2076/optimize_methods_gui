"""
Каталог моделей, алгоритмов подбора и их гиперпараметров.

Пока это заглушки: реальный код моделей ещё не подключён. Чтобы добавить модель,
алгоритм или параметр, достаточно дописать запись в MODELS / ALGORITHMS —
окно подбора построится само.

Источник списка параметров: «Гиперпараметры для серых.md» (pytorch-pointcloud).
В окно вынесены только настраиваемые скалярные параметры; структурные
(in_channels, num_classes, списки каналов и т.п.) задаются архитектурой.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

ParamKind = Literal['float', 'int', 'choice', 'bool']

_TRUE_WORDS = {'true', 'да', 'yes', '1'}
_FALSE_WORDS = {'false', 'нет', 'no', '0'}


@dataclass(frozen=True)
class ParamSpec:
    """Описание одного гиперпараметра: что вводит пользователь и как это проверять."""

    key: str
    label: str
    kind: ParamKind
    default: str
    example: str
    min_value: float | None = None
    max_value: float | None = None
    min_exclusive: bool = False
    choices: tuple[str, ...] = ()

    def parse(self, text: str) -> list[Any]:
        """Разбирает строку «a, b, c» в список значений. При ошибке — ValueError с понятным текстом."""
        if not text.strip():
            raise ValueError('введите хотя бы одно значение.')
        tokens = [token.strip() for token in text.split(',')]
        if any(not token for token in tokens):
            raise ValueError('пустое значение между запятыми.')
        values = [self._parse_token(token) for token in tokens]
        return list(dict.fromkeys(values))  # убираем повторы, сохраняя порядок

    def _parse_token(self, token: str) -> Any:
        if self.kind == 'float':
            try:
                value = float(token)
            except ValueError:
                raise ValueError(f'«{token}» — не число.') from None
            if not math.isfinite(value):
                raise ValueError(f'«{token}» — недопустимое число.')
            self._check_range(token, value)
            return value

        if self.kind == 'int':
            try:
                value = int(token)
            except ValueError:
                raise ValueError(f'«{token}» — не целое число.') from None
            self._check_range(token, value)
            return value

        if self.kind == 'bool':
            word = token.lower()
            if word in _TRUE_WORDS:
                return True
            if word in _FALSE_WORDS:
                return False
            raise ValueError(f'«{token}»: допустимо True или False.')

        word = token.lower()  # choice
        if word not in self.choices:
            raise ValueError(f'«{token}»: допустимо {", ".join(self.choices)}.')
        return word

    def _check_range(self, token: str, value: float) -> None:
        if self.min_value is not None:
            too_small = value < self.min_value or (self.min_exclusive and value == self.min_value)
            if too_small:
                sign = '>' if self.min_exclusive else '≥'
                raise ValueError(f'«{token}»: значение должно быть {sign} {self.min_value:g}.')
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f'«{token}»: значение должно быть ≤ {self.max_value:g}.')


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    description: str
    params: tuple[ParamSpec, ...]


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    name: str
    description: str


def _int(key: str, label: str, default: str, example: str, min_value: int = 1) -> ParamSpec:
    return ParamSpec(key, label, 'int', default, example, min_value=min_value)


def _bool(key: str, label: str, default: str) -> ParamSpec:
    return ParamSpec(key, label, 'bool', default, 'True, False')


def _choice(key: str, label: str, choices: tuple[str, ...], default: str) -> ParamSpec:
    return ParamSpec(key, label, 'choice', default, ', '.join(choices), choices=choices)


TRAINING_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        'learning_rate',
        'Learning rate',
        'float',
        '0.001,'
        ' 0.0001',
        '0.001,'
        ' 0.0001',
        min_value=0,
        min_exclusive=True,
    ),
    _int('epochs', 'Количество эпох (Epochs)', '50, 100, 200', '50, 100, 200'),
    _int('batch_size', 'Размер пакета (Batch size)', '16, 32, 64', '16, 32, 64'),
)

DROPOUT = ParamSpec('dropout', 'Dropout', 'float', '0.2, 0.5', '0.2, 0.5', min_value=0, max_value=1)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key='pointnet2',
        name='PointNet++',
        description='Модель с иерархической обработкой локальных областей',
        params=TRAINING_PARAMS + (
            DROPOUT,
            _choice('sa_aggr', 'Агрегация в SA-блоках (sa_aggr)', ('max', 'mean'), 'max'),
            _bool('use_pos', 'Относительные позиции (use_pos)', 'True'),
            _bool('bias', 'Bias в линейных слоях (bias)', 'False'),
            _bool('act_first', 'Активация до нормализации (act_first)', 'False'),
        ),
    ),
    ModelSpec(
        key='randlanet',
        name='RandLA-Net',
        description='Эффективная модель для больших облаков точек со случайным прореживанием',
        params=TRAINING_PARAMS + (
            DROPOUT,
            _int('decimation', 'Коэффициент прореживания (decimation)', '4', '2, 4'),
            _int('num_neighbors', 'Число соседей (num_neighbors)', '16', '8, 16, 32'),
            _int('stem_channels', 'Каналы stem-слоя (stem_channels)', '8', '8, 16'),
            _bool('bias', 'Bias в линейных слоях (bias)', 'False'),
            _bool('act_first', 'Активация до нормализации (act_first)', 'False'),
        ),
    ),
    ModelSpec(
        key='kpconv',
        name='KPConv',
        description='Конволюционная модель для обработки неупорядоченных точек',
        params=TRAINING_PARAMS + (
            DROPOUT,
            _int('kernel_size', 'Число ядровых точек (kernel_size)', '15', '15'),
            _choice('kp_influence', 'Функция влияния (kp_influence)',
                    ('constant', 'linear', 'gaussian'), 'linear'),
            _choice('fixed_position', 'Фиксированная точка ядра (fixed_position)',
                    ('none', 'center', 'vertical'), 'center'),
            _choice('aggregation_mode', 'Агрегация (aggregation_mode)',
                    ('sum', 'mean', 'max'), 'sum'),
            _choice('stem_type', 'Тип stem-слоя (stem_type)', ('linear', 'kpconv'), 'kpconv'),
            _bool('deformable', 'Deformable-ядро (deformable)', 'False'),
            _bool('modulated', 'Модуляция весов (modulated)', 'False'),
            _bool('bias', 'Bias в KPConv-блоках (bias)', 'False'),
        ),
    ),
)

ALGORITHMS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        'grid',
        'Grid Search (полный перебор)',
        'Будут проверены все возможные комбинации заданных значений.',
    ),
    AlgorithmSpec(
        'random',
        'Random Search (случайный перебор)',
        'Будет проверено случайное подмножество комбинаций заданных значений.',
    ),
    AlgorithmSpec(
        'bayes',
        'Bayesian Optimization (байесовская оптимизация)',
        'Следующие комбинации будут выбираться по результатам предыдущих запусков.',
    ),
)

MODELS_BY_KEY = {model.key: model for model in MODELS}
ALGORITHMS_BY_KEY = {algorithm.key: algorithm for algorithm in ALGORITHMS}


@dataclass(frozen=True)
class SearchConfig:
    """Итог работы окна: что и как подбирать."""

    model: ModelSpec
    algorithm: AlgorithmSpec
    hyperparameters: dict[str, list[Any]]

    @property
    def combinations_count(self) -> int:
        return math.prod(len(values) for values in self.hyperparameters.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            'model': {'key': self.model.key, 'name': self.model.name},
            'algorithm': {'key': self.algorithm.key, 'name': self.algorithm.name},
            'hyperparameters': self.hyperparameters,
            'combinations_count': self.combinations_count,
        }
