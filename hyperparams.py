"""Каталог моделей, алгоритмов подбора и их гиперпараметров.

Пока это заглушки: реальный код моделей ещё не подключён. Чтобы добавить модель,
алгоритм или параметр, достаточно дописать запись в MODELS / ALGORITHMS —
окно подбора построится само.

Источник списка параметров: «Гиперпараметры для серых.md» (pytorch-pointcloud).
Столбцы таблицы в окне берутся из ParamSpec:
    label     → «Назначение»
    key       → «Параметр»
    type_name → «Тип»
    example   → подсказка в пустом поле ввода
    allowed   → «Возможное значение»

Как вводить значения (одно поле — несколько вариантов для перебора):
    • варианты разделяются запятой:              0.001, 0.0001
    • списки пишутся в квадратных скобках:       [64, 128], [128, 256]
      (запятые внутри скобок вариантов не разделяют);
    • None, True и False можно писать в любом регистре;
    • пустое поле — параметр не участвует в подборе.
"""
from __future__ import annotations

import ast
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

ParamKind = Literal['int', 'float', 'bool', 'choice', 'str', 'literal']

_TRUE_WORDS = {'true', 'да', 'yes', '1'}
_FALSE_WORDS = {'false', 'нет', 'no', '0'}
_LITERAL_WORDS = {'true': 'True', 'false': 'False', 'none': 'None', 'null': 'None'}
_NONE_WORDS = {'none', 'null'}
_LEAF_NAMES = {'int': 'целые числа', 'float': 'числа', 'bool': 'значения True / False'}


def split_candidates(text: str) -> list[str]:
    """Делит строку на варианты по запятым, не трогая запятые внутри скобок и кавычек."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote = ''
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = ''
            continue
        if char in '\'"':
            quote = char
        elif char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        elif char == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
            continue
        current.append(char)
    parts.append(''.join(current).strip())
    if depth != 0 or quote:
        raise ValueError('не закрыта скобка или кавычка.')
    return parts


def _to_lists(value: Any) -> Any:
    """Кортежи → списки (рекурсивно), чтобы результат без потерь ложился в JSON."""
    if isinstance(value, (list, tuple)):
        return [_to_lists(item) for item in value]
    return value


def _leaves(value: Any) -> Iterator[Any]:
    if isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _shape_of(value: Any) -> str:
    if value is None:
        return 'none'
    if isinstance(value, dict):
        return 'dict'
    if isinstance(value, list):
        return 'seq'
    return 'scalar'


@dataclass(frozen=True)
class ParamSpec:
    """Описание одного гиперпараметра: строка таблицы и правила проверки ввода."""

    key: str
    label: str
    type_name: str
    allowed: str
    kind: ParamKind
    example: str
    min_value: float | None = None
    max_value: float | None = None
    min_exclusive: bool = False
    choices: tuple[Any, ...] = ()
    nullable: bool = False
    shapes: tuple[str, ...] = ()
    leaf: str | None = None

    def parse(self, text: str) -> list[Any]:
        """
        Разбирает строку в список вариантов. Пустая строка → [] («не задано»).
        При ошибке бросает ValueError с понятным пользователю текстом.
        """
        text = text.strip()
        if not text:
            return []
        tokens = split_candidates(text)
        if any(not token for token in tokens):
            raise ValueError('пустое значение между запятыми.')
        values: list[Any] = []
        for token in tokens:
            value = self._parse_token(token)
            if value not in values:
                values.append(value)
        return values

    def _parse_token(self, token: str) -> Any:
        if self.nullable and token.lower() in _NONE_WORDS:
            return None

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
                int_value = int(token)
            except ValueError:
                raise ValueError(f'«{token}» — не целое число.') from None
            self._check_range(token, int_value)
            return int_value

        if self.kind == 'bool':
            word = token.lower()
            if word in _TRUE_WORDS:
                return True
            if word in _FALSE_WORDS:
                return False
            raise ValueError(f'«{token}»: допустимо True или False.')

        if self.kind == 'choice':
            for choice in self.choices:
                if str(choice).lower() == token.lower():
                    return choice
            raise ValueError(f'«{token}»: допустимо {", ".join(map(str, self.choices))}.')

        if self.kind == 'str':
            return token

        return self._parse_literal(token)

    def _parse_literal(self, token: str) -> Any:
        source = _LITERAL_WORDS.get(token.lower(), token)
        try:
            value = _to_lists(ast.literal_eval(source))
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            raise ValueError(f'«{token}» — не удалось разобрать, пример: {self.example}.') from None

        shape = _shape_of(value)
        if shape not in self.shapes:
            raise ValueError(f'«{token}»: неверный формат, пример: {self.example}.')

        if shape in ('scalar', 'seq') and self.leaf is not None:
            leaves = list(_leaves(value))
            if not leaves:
                raise ValueError(f'«{token}»: пустой список.')
            for leaf in leaves:
                self._check_leaf(token, leaf)
        return value

    def _check_leaf(self, token: str, leaf: Any) -> None:
        if self.leaf == 'bool':
            ok = isinstance(leaf, bool)
        elif self.leaf == 'int':
            ok = isinstance(leaf, int) and not isinstance(leaf, bool)
        else:
            ok = isinstance(leaf, (int, float)) and not isinstance(leaf, bool)
        if ok and isinstance(leaf, float) and not math.isfinite(leaf):
            ok = False
        if not ok:
            raise ValueError(f'«{token}»: ожидаются {_LEAF_NAMES[self.leaf or "float"]}.')
        if self.leaf != 'bool':
            self._check_range(token, leaf)

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


def _literal(
    key: str,
    label: str,
    type_name: str,
    allowed: str,
    example: str,
    shapes: tuple[str, ...],
    leaf: str | None = None,
    **limits: Any,
) -> ParamSpec:
    return ParamSpec(key, label, type_name, allowed, 'literal', example, shapes=shapes, leaf=leaf, **limits)


def _int_seq(
    key: str,
    label: str,
    type_name: str,
    allowed: str,
    example: str,
    min_value: int = 1,
) -> ParamSpec:
    return _literal(key, label, type_name, allowed, example, ('seq',), 'int', min_value=min_value)


def _positive_float_seq(key: str, label: str, type_name: str, allowed: str, example: str) -> ParamSpec:
    return _literal(key, label, type_name, allowed, example, ('seq',), 'float', min_value=0, min_exclusive=True)


def _in_channels() -> ParamSpec:
    return ParamSpec(
        'in_channels',
        'Число входных каналов',
        'int',
        'Любое положительное целое',
        'int',
        '3, 6',
        min_value=1,
    )


def _num_classes(label: str = 'Число классов') -> ParamSpec:
    return ParamSpec(
        'num_classes',
        label,
        'int',
        'Любое положительное целое',
        'int',
        '8',
        min_value=1,
    )


def _spatial_dim() -> ParamSpec:
    return ParamSpec('spatial_dim', 'Размерность координат', 'int', '2 или 3', 'choice', '3', choices=(2, 3))


def _stem_channels(label: str) -> ParamSpec:
    return ParamSpec(
        'stem_channels',
        label,
        'Optional[int]',
        'Целое или None',
        'int',
        '16, None',
        min_value=1,
        nullable=True,
    )


def _aggr_channels(label: str) -> ParamSpec:
    return _literal(
        'aggr_channels',
        label,
        'Optional[Union[int, Sequence[int]]]',
        'Целое, список целых или None',
        '256, [256, 512], None',
        ('scalar', 'seq', 'none'),
        'int',
        min_value=1,
    )


def _dropout(label: str) -> ParamSpec:
    return ParamSpec('dropout', label, 'float', 'Число от 0 до 1', 'float', '0.2, 0.5', min_value=0, max_value=1)


def _act(label: str) -> ParamSpec:
    return ParamSpec(
        'act',
        label,
        'Union[str, Callable, None]',
        'Строка, callable или None',
        'str',
        'relu, leaky_relu',
        nullable=True,
    )


def _norm(label: str) -> ParamSpec:
    return ParamSpec(
        'norm',
        label,
        'Union[str, Callable, None]',
        'Строка, callable или None',
        'str',
        'batch_norm',
        nullable=True,
    )


def _kwargs(key: str, label: str, example: str) -> ParamSpec:
    return _literal(key, label, 'Optional[Dict[str, Any]]', 'Словарь', example, ('dict', 'none'))


def _bool(key: str, label: str, type_name: str = 'bool') -> ParamSpec:
    return ParamSpec(key, label, type_name, 'True / False', 'bool', 'True, False')


def _choice(key: str, label: str, type_name: str, choices: tuple[str, ...]) -> ParamSpec:
    return ParamSpec(
        key,
        label,
        type_name,
        ' или '.join(choices) if len(choices) == 2 else ', '.join(choices),
        'choice',
        ', '.join(choices),
        choices=choices,
    )


TRAINING_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        'learning_rate',
        'Скорость обучения (Learning rate)',
        'float',
        'Число больше 0',
        'float',
        '0.001, 0.0001',
        min_value=0,
        min_exclusive=True,
    ),
    ParamSpec(
        'epochs',
        'Количество эпох (Epochs)',
        'int',
        'Любое положительное целое',
        'int',
        '50, 100, 200',
        min_value=1,
    ),
    ParamSpec(
        'batch_size',
        'Размер пакета (Batch size)',
        'int',
        'Любое положительное целое',
        'int',
        '16, 32, 64',
        min_value=1,
    ),
)

POINTNET2_PARAMS: tuple[ParamSpec, ...] = (
    _in_channels(),
    _num_classes('Число классов сегментации'),
    _stem_channels('Каналы stem внутри энкодера'),
    _int_seq(
        'sa_channels',
        'Каналы SA-блоков',
        'Sequence[Sequence[Union[int, Sequence[int]]]]',
        'См. энкодер',
        '[[32, 32, 64], [64, 64, 128]]',
    ),
    _aggr_channels('Каналы aggregation-MLP после энкодера'),
    _int_seq(
        'fp_channels',
        'Каналы FP-блоков декодера',
        'Sequence[Sequence[int]]',
        'См. декодер',
        '[[256, 256], [128, 128]]',
    ),
    _literal(
        'ratios',
        'Доли сэмплирования SA-блоков',
        'Sequence[float]',
        'Числа от 0 до 1',
        '[0.5, 0.25]',
        ('seq',),
        'float',
        min_value=0,
        max_value=1,
    ),
    _positive_float_seq(
        'radii',
        'Радиусы SA-блоков',
        'Sequence[Union[float, Sequence[float]]]',
        'См. энкодер',
        '[0.1, 0.2]',
    ),
    _int_seq(
        'num_neighbors',
        'Соседи SA-блоков',
        'Sequence[Union[int, Sequence[int]]]',
        'См. энкодер',
        '[16, 32]',
    ),
    _spatial_dim(),
    _act('Активация'),
    _kwargs('act_kwargs', 'Доп. аргументы активации', "{'negative_slope': 0.1}"),
    _bool('act_first', 'Активация до нормализации'),
    _norm('Нормализация'),
    _kwargs('norm_kwargs', 'Доп. аргументы нормализации', "{'momentum': 0.1}"),
    _bool('bias', 'Bias в линейных слоях'),
    _bool('use_pos', 'Конкатенация относительных позиций'),
    _choice('sa_aggr', 'Агрегация в SA-блоках', 'AggrType', ('max', 'mean')),
    _dropout('Dropout перед головой классификации'),
)

RANDLANET_PARAMS: tuple[ParamSpec, ...] = (
    _in_channels(),
    _num_classes(),
    _stem_channels('Каналы stem-MLP (None — пропустить stem)'),
    _int_seq(
        'encoder_channels',
        'Выходные каналы dilated residual блоков',
        'Sequence[int]',
        'Список чётных целых',
        '[16, 64, 128, 256]',
        min_value=2,
    ),
    _int_seq(
        'fp_channels',
        'Каналы декодера на каждом шаге upsampling',
        'Sequence[int]',
        'Список целых',
        '[256, 128, 64, 32]',
    ),
    _int_seq(
        'head_channels',
        'Скрытые каналы MLP сегментационной головы',
        'Sequence[int]',
        'Кортеж/список целых',
        '[64, 32]',
    ),
    _literal(
        'decimation',
        'Коэффициент прореживания между блоками энкодера',
        'Union[int, Sequence[int]]',
        'Целое или последовательность целых',
        '4, [4, 4, 4, 4]',
        ('scalar', 'seq'),
        'int',
        min_value=1,
    ),
    _literal(
        'num_neighbors',
        'Число соседей для K-NN графа в каждом блоке',
        'Union[int, Sequence[int]]',
        'Целое или последовательность целых',
        '16, 32',
        ('scalar', 'seq'),
        'int',
        min_value=1,
    ),
    _aggr_channels('Каналы bottleneck MLP между энкодером и декодером'),
    _dropout('Dropout в сегментационной голове'),
    _act('Активация для декодера и головы'),
    _kwargs('act_kwargs', 'Доп. аргументы активации', "{'negative_slope': 0.1}"),
    _bool('act_first', 'Активация до нормализации'),
    _norm('Нормализация для декодера и головы'),
    _kwargs('norm_kwargs', 'Доп. аргументы нормализации', "{'momentum': 0.1}"),
    _bool('bias', 'Bias в скрытых линейных слоях декодера/головы'),
)

KPCONV_PARAMS: tuple[ParamSpec, ...] = (
    _in_channels(),
    _num_classes(),
    _spatial_dim(),
    _stem_channels('Каналы stem-слоя'),
    _choice(
        'stem_type',
        'Тип stem-слоя',
        "Literal['linear', 'kpconv']",
        ('linear', 'kpconv'),
    ),
    _int_seq(
        'encoder_depths',
        'Число residual-блоков на каждом уровне энкодера',
        'Sequence[int]',
        'Список целых',
        '[1, 1, 2, 2]',
    ),
    _int_seq(
        'encoder_channels',
        'Каналы на каждом уровне энкодера',
        'Sequence[int]',
        'Список целых',
        '[64, 128, 256, 512]',
    ),
    _int_seq(
        'encoder_num_neighbors',
        'Макс. число соседей на каждом уровне энкодера',
        'Sequence[int]',
        'Список целых',
        '[32, 32, 32, 32]',
    ),
    _int_seq(
        'fp_channels',
        'Каналы FP-блоков декодера',
        'Sequence[Sequence[int]]',
        'Список списков целых',
        '[[512, 256], [256, 128]]',
    ),
    _positive_float_seq(
        'grid_sizes',
        'Размеры воксельной сетки для каждого шага downsampling',
        'Sequence[float]',
        'Список положительных чисел',
        '[0.04, 0.08, 0.16]',
    ),
    _positive_float_seq(
        'radii',
        'Радиус поиска соседей на каждом шаге downsampling',
        'Sequence[float]',
        'Список положительных чисел',
        '[0.1, 0.2, 0.4]',
    ),
    ParamSpec(
        'kernel_size',
        'Число ядровых точек K для каждого KPConv-блока',
        'int',
        'Обычно 15',
        'int',
        '15',
        min_value=1,
    ),
    _literal(
        'kp_radius',
        'Радиус сферы ядровых точек на каждом уровне',
        'Union[float, Sequence[float]]',
        'Число или список чисел',
        '0.06, [0.06, 0.12]',
        ('scalar', 'seq'),
        'float',
        min_value=0,
        min_exclusive=True,
    ),
    _literal(
        'kp_sigma',
        'Kernel extent на каждом уровне',
        'Union[float, Sequence[float]]',
        'Число или список чисел', '1.0, [1.0, 1.5]',
        ('scalar', 'seq'),
        'float',
        min_value=0,
        min_exclusive=True,
    ),
    _choice('kp_influence', 'Функция влияния', 'str', ('constant', 'linear', 'gaussian')),
    _choice(
        'fixed_position',
        'Какая ядровая точка фиксирована',
        "Literal['none', 'center', 'vertical']",
        ('none', 'center', 'vertical'),
    ),
    _choice('aggregation_mode', 'Агрегация', 'str', ('sum', 'mean', 'max')),
    _literal(
        'deformable',
        'Использовать ли deformable-ядро на каждом уровне',
        'Union[bool, Sequence]',
        'bool или последовательность bool',
        'False, True',
        ('scalar', 'seq'),
        'bool',
    ),
    _literal(
        'modulated',
        'Использовать ли модуляцию весов',
        'Union[bool, Sequence]',
        'bool или последовательность bool',
        'False, True',
        ('scalar', 'seq'),
        'bool',
    ),
    _norm('Нормализация'),
    _act('Активация'),
    _bool('bias', 'Bias в KPConv-блоках'),
    _dropout('Dropout перед головой классификации'),
)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key='pointnet2',
        name='PointNet++',
        description='Базовая модель для классификации облаков точек',
        params=TRAINING_PARAMS + POINTNET2_PARAMS,
    ),
    ModelSpec(
        key='randlanet',
        name='RandLA-Net',
        description='Модель для семантической сегментации крупномасштабных облаков точек',
        params=TRAINING_PARAMS + RANDLANET_PARAMS,
    ),
    ModelSpec(
        key='kpconv',
        name='KPConv',
        description='Конволюционная модель для обработки неупорядоченных точек',
        params=TRAINING_PARAMS + KPCONV_PARAMS,
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
    """Итог работы окна: что и как подбирать. В hyperparameters только заполненные параметры."""

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
