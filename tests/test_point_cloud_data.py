from pathlib import Path

import numpy as np
import pytest

from point_cloud_canvas import (
    GRADIENTS,
    class_point_scales,
    flight_displacement,
    interpolate_gradient,
    perspective_point_scale,
)
from point_cloud_data import (
    PointCloudLoadError,
    class_style_for,
    load_point_cloud,
    save_point_cloud,
    select_profile_points,
)


def _write_cloud(path: Path, rows: list[str]) -> Path:
    path.write_text('\n'.join(rows), encoding='utf-8')
    return path


def test_load_classified_xyz_preserves_all_points(tmp_path: Path) -> None:
    source = _write_cloud(
        tmp_path / 'classified.xyz',
        [
            '# class X Y Z',
            '2 10 20 30',
            '3 12 24 36',
            '6 14 28 42',
        ],
    )

    cloud = load_point_cloud(source)

    assert cloud.source_count == 3
    assert cloud.rendered_count == 3
    assert cloud.class_counts() == {2: 1, 3: 1, 6: 1}
    np.testing.assert_allclose(cloud.bounds_min, [10, 20, 30])
    np.testing.assert_allclose(cloud.bounds_max, [14, 28, 42])
    np.testing.assert_allclose(cloud.original_points[0], [10, 20, 30])


def test_text_loader_samples_across_the_whole_file(tmp_path: Path) -> None:
    source = _write_cloud(
        tmp_path / 'large.xyz',
        [f'2 {index} 0 0' for index in range(100)],
    )

    cloud = load_point_cloud(source, max_points=10)

    assert cloud.source_count == 100
    assert cloud.rendered_count <= 10
    assert cloud.sampled
    assert cloud.original_points[:, 0].min() == 0
    assert cloud.original_points[:, 0].max() >= 85


def test_text_loader_keeps_every_point_by_default(tmp_path: Path) -> None:
    source = _write_cloud(
        tmp_path / 'complete.xyz',
        [f'2 {index} 0 0' for index in range(100)],
    )

    cloud = load_point_cloud(source)

    assert cloud.source_count == 100
    assert cloud.rendered_count == 100
    assert not cloud.sampled


def test_loader_skips_malformed_rows_and_reads_rgb(tmp_path: Path) -> None:
    source = _write_cloud(
        tmp_path / 'rgb.xyz',
        [
            '2 1 2 3 255 128 0',
            'broken row',
            '3 4 5 6 0 64 255',
        ],
    )

    cloud = load_point_cloud(source)

    assert cloud.invalid_count == 1
    assert cloud.rgb is not None
    np.testing.assert_allclose(cloud.rgb[0], [1.0, 128 / 255, 0.0])


@pytest.mark.parametrize('extension', ['.xyz', '.las', '.laz'])
def test_export_round_trip(tmp_path: Path, extension: str) -> None:
    source = _write_cloud(
        tmp_path / 'source.xyz',
        [
            '2 10.125 20.25 30.5 255 128 0',
            '6 11.75 21.5 31.25 0 64 255',
        ],
    )
    cloud = load_point_cloud(source)

    exported = save_point_cloud(cloud, tmp_path / f'converted{extension}')
    restored = load_point_cloud(exported)

    assert restored.class_counts() == {2: 1, 6: 1}
    np.testing.assert_allclose(
        restored.original_points,
        cloud.original_points,
        atol=0.001,
    )
    assert restored.rgb is not None
    np.testing.assert_allclose(restored.rgb, cloud.rgb, atol=1 / 255)


def test_profile_selects_only_corridor_points(tmp_path: Path) -> None:
    source = _write_cloud(
        tmp_path / 'profile.xyz',
        [
            '2 0 0 1',
            '2 5 0.4 2',
            '3 8 3 3',
            '3 12 0 4',
        ],
    )
    cloud = load_point_cloud(source)

    distance, heights, classes = select_profile_points(
        cloud,
        np.asarray([0, 0, 0]),
        np.asarray([10, 0, 0]),
        width=1.0,
    )

    np.testing.assert_allclose(distance, [0, 5])
    np.testing.assert_allclose(heights, [1, 2])
    np.testing.assert_array_equal(classes, [2, 2])


def test_unknown_class_gets_stable_fallback_style() -> None:
    first = class_style_for(99)
    second = class_style_for(99)

    assert first == second
    assert first.title == 'Класс 99'
    assert first.hex_color.startswith('#')


def test_smartlaser_class_palette_is_preserved() -> None:
    assert class_style_for(2).color == (224, 122, 95)
    assert class_style_for(6).color == (226, 6, 44)
    assert class_style_for(16).color == (212, 241, 249)
    assert class_style_for(21).color == (255, 255, 255)
    assert class_style_for(22).color == (212, 241, 249)


def test_gradient_uses_original_non_uniform_stop_positions() -> None:
    np.testing.assert_allclose(
        interpolate_gradient('TURBO', np.asarray([0.07], dtype=np.float32))[0],
        GRADIENTS['TURBO'][1, 1:],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        interpolate_gradient('GRAYSCALE', np.asarray([0.25], dtype=np.float32))[0],
        [0.25, 0.25, 0.25],
        atol=1e-6,
    )


def test_class_size_and_multiplier_are_both_applied() -> None:
    scales = class_point_scales(
        np.asarray([2, 2, 6, 9], dtype=np.int32),
        sizes={2: 0.3, 6: 0.7, 9: 2.0},
        multipliers={2: 2.0, 6: 0.5, 9: 3.0},
    )

    np.testing.assert_allclose(scales, [0.6, 0.6, 0.35, 6.0])


def test_perspective_point_scale_uses_viewport_and_fov() -> None:
    base = perspective_point_scale(720, 1.0, 75.0)
    retina = perspective_point_scale(720, 2.0, 75.0)
    wide_angle = perspective_point_scale(720, 1.0, 100.0)

    assert retina == pytest.approx(base * 2.0)
    assert wide_angle < base


def test_flight_displacement_has_constant_diagonal_speed() -> None:
    movement = flight_displacement(
        forward=np.asarray([1.0, 0.0, 0.0]),
        right=np.asarray([0.0, 1.0, 0.0]),
        forward_axis=1.0,
        side_axis=1.0,
        vertical_axis=1.0,
        speed=12.0,
        elapsed=0.5,
    )

    assert np.linalg.norm(movement) == pytest.approx(6.0)
    assert all(component > 0.0 for component in movement)


def test_flight_displacement_supports_vertical_descent() -> None:
    movement = flight_displacement(
        forward=np.asarray([1.0, 0.0, 0.0]),
        right=np.asarray([0.0, 1.0, 0.0]),
        forward_axis=0.0,
        side_axis=0.0,
        vertical_axis=-1.0,
        speed=4.0,
        elapsed=0.25,
    )

    np.testing.assert_allclose(movement, [0.0, 0.0, -1.0])


def test_unsupported_file_is_rejected(tmp_path: Path) -> None:
    source = _write_cloud(tmp_path / 'cloud.csv', ['2 1 2 3'])

    with pytest.raises(PointCloudLoadError, match='XYZ, TXT, PTS, LAS и LAZ'):
        load_point_cloud(source)
