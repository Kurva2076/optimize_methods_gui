from __future__ import annotations

import colorsys
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ProgressCallback = Callable[[int], None]
CancelCallback = Callable[[], bool]

SUPPORTED_POINT_CLOUD_EXTENSIONS = {'.xyz', '.txt', '.pts', '.las', '.laz'}


class PointCloudLoadError(ValueError):
    """Ошибка чтения облака точек, которую можно показать пользователю."""


class PointCloudLoadCancelled(RuntimeError):
    """Загрузка была отменена пользователем."""


@dataclass(frozen=True, slots=True)
class ClassStyle:
    class_id: int
    title: str
    color: tuple[int, int, int]

    @property
    def hex_color(self) -> str:
        return '#{:02X}{:02X}{:02X}'.format(*self.color)


# Палитра и названия повторяют классы SmartLaser Point Viewer. Для интерфейса
# названия переведены, а номера оставлены по спецификации ASPRS/LAS.
CLASS_STYLES: dict[int, ClassStyle] = {
    0: ClassStyle(0, 'Неклассифицированные', (34, 34, 59)),
    1: ClassStyle(1, 'По умолчанию', (74, 78, 105)),
    2: ClassStyle(2, 'Грунт', (224, 122, 95)),
    3: ClassStyle(3, 'Низкая растительность', (45, 106, 79)),
    4: ClassStyle(4, 'Средняя растительность', (116, 198, 157)),
    5: ClassStyle(5, 'Высокая растительность', (149, 213, 178)),
    6: ClassStyle(6, 'Строения и крыши', (226, 6, 44)),
    7: ClassStyle(7, 'Низкий шум', (114, 9, 183)),
    8: ClassStyle(8, 'Служебные точки A', (242, 233, 228)),
    9: ClassStyle(9, 'Вода', (212, 241, 249)),
    10: ClassStyle(10, 'Железная дорога', (207, 74, 55)),
    11: ClassStyle(11, 'Дорожное покрытие', (183, 177, 174)),
    12: ClassStyle(12, 'Служебные точки B', (242, 233, 228)),
    13: ClassStyle(13, 'Защитный провод', (136, 23, 26)),
    14: ClassStyle(14, 'Фазный провод', (136, 23, 26)),
    15: ClassStyle(15, 'Опора ЛЭП', (84, 76, 80)),
    16: ClassStyle(16, 'Соединитель проводов', (212, 241, 249)),
    17: ClassStyle(17, 'Мостовое полотно', (212, 241, 249)),
    18: ClassStyle(18, 'Высокий шум', (212, 241, 249)),
    19: ClassStyle(19, 'Надземная конструкция', (212, 241, 249)),
    20: ClassStyle(20, 'Игнорируемый грунт', (212, 241, 249)),
    21: ClassStyle(21, 'Снег', (255, 255, 255)),
    22: ClassStyle(22, 'Временное исключение', (212, 241, 249)),
}


def class_style_for(class_id: int) -> ClassStyle:
    """Возвращает стабильный цвет и читаемое имя для любого номера класса."""
    if class_id in CLASS_STYLES:
        return CLASS_STYLES[class_id]

    hue = ((class_id * 0.61803398875) % 1.0)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.88)
    color = tuple(round(channel * 255) for channel in (red, green, blue))
    return ClassStyle(class_id, f'Класс {class_id}', color)


@dataclass(slots=True)
class PointCloudData:
    path: Path
    points: np.ndarray
    classes: np.ndarray
    center: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    source_count: int
    invalid_count: int = 0
    rgb: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError('points должен иметь форму (N, 3)')
        if self.classes.shape != (len(self.points),):
            raise ValueError('classes должен содержать один класс на точку')
        if self.rgb is not None and self.rgb.shape != self.points.shape:
            raise ValueError('rgb должен иметь форму (N, 3)')

    @property
    def rendered_count(self) -> int:
        return len(self.points)

    @property
    def sampled(self) -> bool:
        return self.rendered_count < self.source_count

    @property
    def original_points(self) -> np.ndarray:
        return self.points.astype(np.float64, copy=False) + self.center

    @property
    def unique_classes(self) -> np.ndarray:
        return np.unique(self.classes)

    def class_counts(self) -> dict[int, int]:
        ids, counts = np.unique(self.classes, return_counts=True)
        return {
            int(class_id): int(count)
            for class_id, count in zip(ids, counts, strict=True)
        }


def load_point_cloud(
    path: str | Path,
    *,
    max_points: int | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> PointCloudData:
    """Загружает облако точек целиком либо с заданным пределом отображения.

    Текстовые файлы читаются последовательно. При заданном ``max_points``
    выбирается равномерная детерминированная выборка по всему файлу.
    """
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f'Файл не найден: {file_path}')
    if file_path.suffix.lower() not in SUPPORTED_POINT_CLOUD_EXTENSIONS:
        raise PointCloudLoadError(
            'Поддерживаются файлы XYZ, TXT, PTS, LAS и LAZ.'
        )
    if max_points is not None and max_points < 1:
        raise ValueError('max_points должен быть положительным')

    progress_callback = progress or (lambda _value: None)
    cancel_callback = cancelled or (lambda: False)
    progress_callback(0)

    if file_path.suffix.lower() in {'.las', '.laz'}:
        result = _load_las(
            file_path,
            max_points=max_points,
            progress=progress_callback,
            cancelled=cancel_callback,
        )
    else:
        result = _load_text(
            file_path,
            max_points=max_points,
            progress=progress_callback,
            cancelled=cancel_callback,
        )

    progress_callback(100)
    return result


def save_point_cloud(cloud: PointCloudData, path: str | Path) -> Path:
    """Сохраняет загруженные точки в XYZ/TXT или LAS/LAZ.

    Если исходный файл был прорежен для отображения, экспорт содержит именно
    отображаемую равномерную выборку. Диалог просмотрщика отдельно предупреждает
    пользователя об этом до записи.
    """
    target = Path(path).expanduser().resolve()
    extension = target.suffix.lower()
    if extension not in {'.xyz', '.txt', '.las', '.laz'}:
        raise PointCloudLoadError(
            'Экспорт поддерживает XYZ, TXT, LAS и LAZ.'
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    original = cloud.original_points
    if extension in {'.xyz', '.txt'}:
        columns: list[np.ndarray] = [cloud.classes.reshape(-1, 1), original]
        format_specifiers = ['%d', '%.6f', '%.6f', '%.6f']
        if cloud.rgb is not None:
            rgb = np.rint(np.clip(cloud.rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            columns.append(rgb)
            format_specifiers.extend(('%d', '%d', '%d'))
        np.savetxt(
            target,
            np.column_stack(columns),
            fmt=format_specifiers,
            delimiter=' ',
        )
        return target

    try:
        import laspy
    except ImportError as error:
        raise PointCloudLoadError(
            'Для LAS/LAZ установите зависимости из requirements.txt.'
        ) from error

    point_format = 3 if cloud.rgb is not None else 1
    header = laspy.LasHeader(point_format=point_format, version='1.2')
    header.scales = np.asarray((0.001, 0.001, 0.001))
    header.offsets = np.floor(cloud.bounds_min)
    las = laspy.LasData(header)
    las.x = original[:, 0]
    las.y = original[:, 1]
    las.z = original[:, 2]
    las.classification = np.clip(cloud.classes, 0, 255).astype(np.uint8)
    if cloud.rgb is not None:
        rgb16 = np.rint(np.clip(cloud.rgb, 0.0, 1.0) * 65535.0).astype(np.uint16)
        las.red = rgb16[:, 0]
        las.green = rgb16[:, 1]
        las.blue = rgb16[:, 2]
    try:
        las.write(target)
    except Exception as error:
        raise PointCloudLoadError(
            f'Не удалось сохранить {extension.upper()[1:]}: {error}'
        ) from error
    return target


def _estimate_text_stride(path: Path, max_points: int | None) -> int:
    if max_points is None:
        return 1
    sample_size = min(path.stat().st_size, 512 * 1024)
    with path.open('rb') as stream:
        sample = stream.read(sample_size)

    lines = [line for line in sample.splitlines() if line.strip()]
    if not lines:
        return 1

    average_line_size = max(1.0, sum(len(line) + 1 for line in lines) / len(lines))
    estimated_rows = max(1, round(path.stat().st_size / average_line_size))
    return max(1, math.ceil(estimated_rows / max_points))


def _text_schema(column_count: int) -> tuple[slice, int | None, slice | None]:
    if column_count == 3:
        return slice(0, 3), None, None
    if column_count == 4:
        return slice(1, 4), 0, None
    if column_count == 6:
        return slice(0, 3), None, slice(3, 6)
    if column_count == 7:
        return slice(1, 4), 0, slice(4, 7)
    if column_count == 8:
        return slice(1, 4), 0, slice(5, 8)
    raise PointCloudLoadError(
        'Неверный формат строки. Ожидались X Y Z, class X Y Z '
        'или координаты с RGB.'
    )


def _load_text(
    path: Path,
    *,
    max_points: int | None,
    progress: ProgressCallback,
    cancelled: CancelCallback,
) -> PointCloudData:
    file_size = max(1, path.stat().st_size)
    stride = _estimate_text_stride(path, max_points)
    source_count = 0
    invalid_count = 0
    schema_columns: int | None = None
    selected_lines: list[bytes] = []
    data_chunks: list[np.ndarray] = []
    bytes_read = 0
    last_progress = -1

    def flush_lines() -> None:
        nonlocal invalid_count
        if not selected_lines:
            return
        normalized = b'\n'.join(selected_lines).replace(b',', b'')
        raw = np.fromstring(normalized.decode('ascii', errors='ignore'), sep=' ')
        expected = len(selected_lines) * int(schema_columns or 0)
        if raw.size != expected:
            # Редкий путь для повреждённого файла: сохраняем только валидные строки.
            for line in selected_lines:
                values = np.fromstring(
                    line.replace(b',', b'').decode('ascii', errors='ignore'),
                    sep=' ',
                )
                if values.size == schema_columns:
                    data_chunks.append(values.reshape(1, -1))
                else:
                    invalid_count += 1
        else:
            data_chunks.append(raw.reshape(-1, int(schema_columns)))
        selected_lines.clear()

    with path.open('rb') as stream:
        for raw_line in stream:
            if cancelled():
                raise PointCloudLoadCancelled('Загрузка отменена')

            bytes_read += len(raw_line)
            stripped = raw_line.strip()
            if not stripped or stripped.startswith(b'#'):
                continue

            columns = len(stripped.replace(b',', b'').split())
            if columns == 1 and source_count == 0:
                # В PTS первая строка иногда содержит только количество точек.
                continue

            current_index = source_count
            source_count += 1
            if current_index % stride != 0:
                continue

            if schema_columns is None:
                _text_schema(columns)
                schema_columns = columns
            if columns != schema_columns:
                invalid_count += 1
                continue

            selected_lines.append(stripped)
            if len(selected_lines) >= 25_000:
                flush_lines()

            current_progress = min(96, int(bytes_read / file_size * 96))
            if current_progress != last_progress:
                progress(current_progress)
                last_progress = current_progress

    flush_lines()
    if not data_chunks or source_count == 0 or schema_columns is None:
        raise PointCloudLoadError('Файл не содержит корректных точек.')

    data = np.concatenate(data_chunks, axis=0)
    if max_points is not None and len(data) > max_points:
        indices = np.linspace(0, len(data) - 1, max_points, dtype=np.int64)
        data = data[indices]

    coordinates_slice, class_column, rgb_slice = _text_schema(schema_columns)
    points = data[:, coordinates_slice]
    classes = (
        np.zeros(len(data), dtype=np.int16)
        if class_column is None
        else np.rint(data[:, class_column]).astype(np.int16)
    )
    rgb = None if rgb_slice is None else _normalize_rgb(data[:, rgb_slice])
    return _finalize_cloud(
        path,
        points,
        classes,
        source_count=source_count,
        invalid_count=invalid_count,
        rgb=rgb,
    )


def _load_las(
    path: Path,
    *,
    max_points: int | None,
    progress: ProgressCallback,
    cancelled: CancelCallback,
) -> PointCloudData:
    try:
        import laspy
    except ImportError as error:
        raise PointCloudLoadError(
            'Для LAS/LAZ установите зависимости из requirements.txt.'
        ) from error

    point_chunks: list[np.ndarray] = []
    class_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []

    try:
        with laspy.open(path) as reader:
            source_count = int(reader.header.point_count)
            if source_count == 0:
                raise PointCloudLoadError('LAS/LAZ-файл не содержит точек.')
            stride = (
                1
                if max_points is None
                else max(1, math.ceil(source_count / max_points))
            )
            offset = 0
            has_rgb = all(
                name in reader.header.point_format.dimension_names
                for name in ('red', 'green', 'blue')
            )

            for chunk in reader.chunk_iterator(500_000):
                if cancelled():
                    raise PointCloudLoadCancelled('Загрузка отменена')

                count = len(chunk)
                indices = np.arange(offset, offset + count, dtype=np.int64)
                mask = indices % stride == 0
                point_chunks.append(
                    np.column_stack((chunk.x[mask], chunk.y[mask], chunk.z[mask]))
                )
                class_chunks.append(
                    np.asarray(chunk.classification[mask], dtype=np.int16)
                )
                if has_rgb:
                    rgb_chunks.append(
                        np.column_stack(
                            (chunk.red[mask], chunk.green[mask], chunk.blue[mask])
                        )
                    )
                offset += count
                progress(min(96, int(offset / source_count * 96)))
    except PointCloudLoadCancelled:
        raise
    except PointCloudLoadError:
        raise
    except Exception as error:
        raise PointCloudLoadError(f'Не удалось прочитать LAS/LAZ: {error}') from error

    points = np.concatenate(point_chunks, axis=0)
    classes = np.concatenate(class_chunks, axis=0)
    rgb = _normalize_rgb(np.concatenate(rgb_chunks, axis=0)) if rgb_chunks else None
    return _finalize_cloud(
        path,
        points,
        classes,
        source_count=source_count,
        invalid_count=0,
        rgb=rgb,
    )


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    result = np.asarray(rgb, dtype=np.float32)
    if result.size and float(np.nanmax(result)) > 1.0:
        divisor = 65535.0 if float(np.nanmax(result)) > 255.0 else 255.0
        result /= divisor
    return np.clip(result, 0.0, 1.0)


def _finalize_cloud(
    path: Path,
    points: np.ndarray,
    classes: np.ndarray,
    *,
    source_count: int,
    invalid_count: int,
    rgb: np.ndarray | None,
) -> PointCloudData:
    points64 = np.asarray(points, dtype=np.float64)
    finite_mask = np.isfinite(points64).all(axis=1)
    invalid_count += int((~finite_mask).sum())
    points64 = points64[finite_mask]
    classes = np.asarray(classes, dtype=np.int16)[finite_mask]
    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.float32)[finite_mask]

    if not len(points64):
        raise PointCloudLoadError('После проверки в файле не осталось корректных точек.')

    bounds_min = points64.min(axis=0)
    bounds_max = points64.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0
    centered_points = (points64 - center).astype(np.float32)

    return PointCloudData(
        path=path,
        points=centered_points,
        classes=classes,
        center=center,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        source_count=source_count,
        invalid_count=invalid_count,
        rgb=rgb,
    )


def select_profile_points(
    cloud: PointCloudData,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    *,
    max_points: int = 120_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Выбирает точки в вертикальном коридоре и строит профиль расстояние–Z."""
    start_xy = np.asarray(start, dtype=np.float64)[:2]
    end_xy = np.asarray(end, dtype=np.float64)[:2]
    direction = end_xy - start_xy
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        raise ValueError('Для профиля выберите две разные точки.')
    unit = direction / length

    original = cloud.original_points
    relative = original[:, :2] - start_xy
    along = relative @ unit
    perpendicular = np.abs(relative[:, 0] * unit[1] - relative[:, 1] * unit[0])
    mask = (
        (along >= 0.0)
        & (along <= length)
        & (perpendicular <= max(width, 1e-9) / 2.0)
    )

    selected_along = along[mask]
    selected_z = original[mask, 2]
    selected_classes = cloud.classes[mask]
    if len(selected_along) > max_points:
        indices = np.linspace(0, len(selected_along) - 1, max_points, dtype=np.int64)
        selected_along = selected_along[indices]
        selected_z = selected_z[indices]
        selected_classes = selected_classes[indices]
    return selected_along, selected_z, selected_classes
