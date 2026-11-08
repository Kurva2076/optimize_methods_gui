from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFocusEvent,
    QFont,
    QKeyEvent,
    QMatrix4x4,
    QMouseEvent,
    QOpenGLFunctions,
    QPainter,
    QPen,
    QVector2D,
    QVector3D,
    QWheelEvent,
)
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLShader,
    QOpenGLShaderProgram,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from point_cloud_data import PointCloudData, class_style_for

GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_DEPTH_TEST = 0x0B71
GL_BLEND = 0x0BE2
GL_PROGRAM_POINT_SIZE = 0x8642
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_FLOAT = 0x1406
GL_LINES = 0x0001
GL_POINTS = 0x0000
GL_TRIANGLE_STRIP = 0x0005
GL_TEXTURE0 = 0x84C0
GL_TEXTURE_2D = 0x0DE1
GL_FRAMEBUFFER = 0x8D40
GL_LESS = 0x0201
GL_LEQUAL = 0x0203

FLIGHT_KEYS = frozenset(
    {
        Qt.Key.Key_W,
        Qt.Key.Key_A,
        Qt.Key.Key_S,
        Qt.Key.Key_D,
        Qt.Key.Key_Space,
        Qt.Key.Key_Shift,
    }
)


VERTEX_SHADER = """
#version 120
attribute vec3 position;
attribute vec3 color;
attribute float pointScale;
uniform mat4 mvp;
uniform float pointSize;
uniform float perspectiveScale;
uniform bool sizeAttenuation;
varying vec3 vertexColor;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    float attenuation = sizeAttenuation
        ? clamp(perspectiveScale / max(gl_Position.w, 0.001), 0.35, 12.0)
        : 1.0;
    gl_PointSize = pointSize * pointScale * attenuation;
    vertexColor = color;
}
"""


FRAGMENT_SHADER = """
#version 120
varying vec3 vertexColor;
uniform bool roundPoints;
uniform bool pointPass;
uniform bool overrideColor;
uniform vec3 solidColor;

void main() {
    if (pointPass && roundPoints) {
        vec2 delta = gl_PointCoord - vec2(0.5);
        if (dot(delta, delta) > 0.25) {
            discard;
        }
    }
    float alpha = pointPass ? 0.96 : 0.55;
    vec3 outputColor = overrideColor ? solidColor : vertexColor;
    gl_FragColor = vec4(outputColor, alpha);
}
"""


SCREEN_VERTEX_SHADER = """
#version 120
attribute vec2 position;
varying vec2 textureCoordinate;

void main() {
    textureCoordinate = (position + vec2(1.0)) * 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""


SOBEL_FRAGMENT_SHADER = """
#version 120
uniform sampler2D sceneTexture;
uniform vec2 texelSize;
varying vec2 textureCoordinate;

float luminance(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

void main() {
    mat3 gx = mat3(
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0
    );
    mat3 gy = mat3(
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0
    );
    mat3 sampleValue;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            sampleValue[x + 1][y + 1] = luminance(
                texture2D(sceneTexture, textureCoordinate + offset).rgb
            );
        }
    }
    float edgeX = dot(gx[0], sampleValue[0])
        + dot(gx[1], sampleValue[1])
        + dot(gx[2], sampleValue[2]);
    float edgeY = dot(gy[0], sampleValue[0])
        + dot(gy[1], sampleValue[1])
        + dot(gy[2], sampleValue[2]);
    float edge = sqrt(edgeX * edgeX + edgeY * edgeY);
    vec3 sourceColor = texture2D(sceneTexture, textureCoordinate).rgb;
    gl_FragColor = vec4(edge > 0.1 ? vec3(edge) : sourceColor, 1.0);
}
"""


BACKGROUND_FRAGMENT_SHADER = """
#version 120
uniform vec3 topColor;
uniform vec3 bottomColor;
varying vec2 textureCoordinate;

void main() {
    float glow = smoothstep(0.0, 1.0, textureCoordinate.y);
    vec3 color = mix(bottomColor, topColor, glow);
    gl_FragColor = vec4(color, 1.0);
}
"""


# Позиции и RGB-значения воспроизводят gradient.js из SmartLaser Point Viewer.
# Первая колонка — положение цветовой остановки в диапазоне 0..1.
GRADIENTS: dict[str, np.ndarray] = {
    'SPECTRAL': np.asarray(
        [
            (0.0, 0.3686, 0.3098, 0.6353),
            (0.1, 0.1961, 0.5333, 0.7412),
            (0.2, 0.4000, 0.7608, 0.6471),
            (0.3, 0.6706, 0.8667, 0.6431),
            (0.4, 0.9020, 0.9608, 0.5961),
            (0.5, 1.0000, 1.0000, 0.7490),
            (0.6, 0.9961, 0.8784, 0.5451),
            (0.7, 0.9922, 0.6824, 0.3804),
            (0.8, 0.9569, 0.4275, 0.2627),
            (0.9, 0.8353, 0.2431, 0.3098),
            (1.0, 0.6196, 0.0039, 0.2588),
        ],
        dtype=np.float32,
    ),
    'PLASMA': np.asarray(
        [
            (0.0, 0.241, 0.015, 0.610),
            (0.1, 0.387, 0.001, 0.654),
            (0.2, 0.524, 0.025, 0.653),
            (0.3, 0.651, 0.125, 0.596),
            (0.4, 0.752, 0.227, 0.513),
            (0.5, 0.837, 0.329, 0.431),
            (0.6, 0.907, 0.435, 0.353),
            (0.7, 0.963, 0.554, 0.272),
            (0.8, 0.992, 0.681, 0.195),
            (0.9, 0.987, 0.822, 0.144),
            (1.0, 0.940, 0.975, 0.131),
        ],
        dtype=np.float32,
    ),
    'YELLOW_GREEN': np.asarray(
        [
            (0.0, 0.1647, 0.2824, 0.3451),
            (0.1, 0.1338, 0.3555, 0.4227),
            (0.2, 0.0610, 0.4319, 0.4864),
            (0.3, 0.0000, 0.5099, 0.5319),
            (0.4, 0.0000, 0.5881, 0.5569),
            (0.5, 0.1370, 0.6650, 0.5614),
            (0.6, 0.2906, 0.7395, 0.5477),
            (0.7, 0.4453, 0.8099, 0.5201),
            (0.8, 0.6102, 0.8748, 0.4850),
            (0.9, 0.7883, 0.9323, 0.4514),
            (1.0, 0.9804, 0.9804, 0.4314),
        ],
        dtype=np.float32,
    ),
    'VIRIDIS': np.asarray(
        [
            (0.0, 0.267, 0.005, 0.329),
            (0.1, 0.283, 0.141, 0.458),
            (0.2, 0.254, 0.265, 0.530),
            (0.3, 0.207, 0.372, 0.553),
            (0.4, 0.164, 0.471, 0.558),
            (0.5, 0.128, 0.567, 0.551),
            (0.6, 0.135, 0.659, 0.518),
            (0.7, 0.267, 0.749, 0.441),
            (0.8, 0.478, 0.821, 0.318),
            (0.9, 0.741, 0.873, 0.150),
            (1.0, 0.993, 0.906, 0.144),
        ],
        dtype=np.float32,
    ),
    'INFERNO': np.asarray(
        [
            (0.0, 0.077, 0.042, 0.206),
            (0.1, 0.225, 0.036, 0.388),
            (0.2, 0.373, 0.074, 0.432),
            (0.3, 0.522, 0.128, 0.420),
            (0.4, 0.665, 0.182, 0.370),
            (0.5, 0.797, 0.255, 0.287),
            (0.6, 0.902, 0.364, 0.184),
            (0.7, 0.969, 0.516, 0.063),
            (0.8, 0.988, 0.683, 0.072),
            (0.9, 0.961, 0.859, 0.298),
            (1.0, 0.988, 0.998, 0.645),
        ],
        dtype=np.float32,
    ),
    'GRAYSCALE': np.asarray(
        ((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)),
        dtype=np.float32,
    ),
    'TURBO': np.asarray(
        [
            (0.00, 0.18995, 0.07176, 0.23217),
            (0.07, 0.25107, 0.25237, 0.63374),
            (0.13, 0.27628, 0.42118, 0.89123),
            (0.20, 0.25862, 0.57958, 0.99876),
            (0.27, 0.15844, 0.73551, 0.92305),
            (0.33, 0.09267, 0.86554, 0.76230),
            (0.40, 0.19659, 0.94901, 0.59466),
            (0.47, 0.42778, 0.99419, 0.38575),
            (0.53, 0.64362, 0.98999, 0.23356),
            (0.60, 0.80473, 0.92452, 0.20459),
            (0.67, 0.93301, 0.81236, 0.22667),
            (0.73, 0.99314, 0.67408, 0.20348),
            (0.80, 0.98360, 0.49291, 0.12849),
            (0.87, 0.92105, 0.31489, 0.05475),
            (0.93, 0.81608, 0.18462, 0.01809),
            (1.00, 0.66449, 0.08436, 0.00424),
        ],
        dtype=np.float32,
    ),
    'RAINBOW': np.asarray(
        [
            (0.0, 0.278, 0.00, 0.714),
            (1 / 6, 0.00, 0.00, 1.00),
            (2 / 6, 0.00, 1.00, 1.00),
            (3 / 6, 0.00, 1.00, 0.00),
            (4 / 6, 1.00, 1.00, 0.00),
            (5 / 6, 1.00, 0.64, 0.00),
            (1.0, 1.00, 0.00, 0.00),
        ],
        dtype=np.float32,
    ),
    'CONTOUR': np.asarray(
        [
            (0.00, 0.0, 0.0, 0.0),
            (0.03, 0.0, 0.0, 0.0),
            (0.04, 1.0, 1.0, 1.0),
            (1.00, 1.0, 1.0, 1.0),
        ],
        dtype=np.float32,
    ),
}


def interpolate_gradient(name: str, normalized: np.ndarray) -> np.ndarray:
    """Интерполировать RGB по исходным позициям остановок SmartLaser."""
    if name not in GRADIENTS:
        raise ValueError(f'Неизвестный градиент: {name}')
    values = np.asarray(normalized, dtype=np.float32)
    gradient = GRADIENTS[name]
    positions = gradient[:, 0]
    colors = gradient[:, 1:]
    result = np.empty((len(values), 3), dtype=np.float32)
    for channel in range(3):
        result[:, channel] = np.interp(
            values,
            positions,
            colors[:, channel],
        )
    return result


def class_point_scales(
    classes: np.ndarray,
    sizes: dict[int, float],
    multipliers: dict[int, float],
) -> np.ndarray:
    """Вернуть эффективный размер: абсолютный размер × множитель класса."""
    result = np.ones(len(classes), dtype=np.float32)
    for class_id in np.unique(classes):
        value = int(class_id)
        result[classes == class_id] = sizes.get(value, 0.15) * multipliers.get(
            value,
            1.0,
        )
    return result


def perspective_point_scale(
    viewport_height: int,
    pixel_ratio: float,
    field_of_view: float,
) -> float:
    """Масштаб перспективных точек по формуле Three.js PointsMaterial."""
    physical_height = max(1.0, float(viewport_height) * max(1.0, pixel_ratio))
    half_fov = math.radians(float(field_of_view)) / 2.0
    return physical_height / (2.0 * math.tan(half_fov))


def flight_displacement(
    forward: np.ndarray,
    right: np.ndarray,
    forward_axis: float,
    side_axis: float,
    vertical_axis: float,
    speed: float,
    elapsed: float,
) -> np.ndarray:
    """Рассчитать перемещение свободной камеры с постоянной скоростью."""
    movement = (
        np.asarray(forward, dtype=np.float64) * forward_axis
        + np.asarray(right, dtype=np.float64) * side_axis
        + np.asarray([0.0, 0.0, 1.0]) * vertical_axis
    )
    length = float(np.linalg.norm(movement))
    if length <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    return movement / length * max(0.0, speed) * max(0.0, elapsed)


class PointCloudCanvas(QOpenGLWidget):
    """Встроенный интерактивный OpenGL-просмотрщик облака точек."""

    point_picked = Signal(object)
    measurement_changed = Signal(str)
    profile_ready = Signal(object, object)
    camera_changed = Signal(str)
    render_error = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName('pointCloudCanvas')
        self.setMinimumSize(620, 480)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.cloud: PointCloudData | None = None
        self.visible_classes: set[int] = set()
        self.class_colors: dict[int, QColor] = {}
        self.class_sizes: dict[int, float] = {}
        self.class_multipliers: dict[int, float] = {}
        self.color_mode = 'classification'
        self.gradient_name = 'RAINBOW'
        self.round_points = True
        self.point_size = 1.0
        self.edge_detection = False
        self.size_attenuation = True
        self.marker_color = QColor('#FFFF00')
        self.line_color = QColor('#3B82F6')
        self.perspective = True
        self.field_of_view = 75.0
        self.navigation_mode = 'orbit'
        self.flight_speed = 10.0
        self.show_grid = True
        self.gradient_background = False
        self.clip_enabled = False
        self.clip_min = -math.inf
        self.clip_max = math.inf
        self.profile_clip_mode = 'none'
        self.profile_clip_enabled = False
        self.profile_helpers_visible = False
        self.profile_width = 2.0
        self._profile_segment: tuple[np.ndarray, np.ndarray] | None = None

        self.yaw = math.radians(-42.0)
        self.pitch = math.radians(34.0)
        self.distance = 10.0
        self.target = np.zeros(3, dtype=np.float64)
        self._scene_radius = 1.0

        self.tool_mode = 'camera'
        self._ruler_paths: list[list[np.ndarray]] = [[]]
        self._profile_points: list[np.ndarray] = []
        self._info_points: list[tuple[np.ndarray, str]] = []
        self._hover_point: np.ndarray | None = None
        self._hover_label = ''
        self._last_hover_pick = 0.0
        self._last_mouse_position = QPoint()
        self._drag_button = Qt.MouseButton.NoButton
        self._angular_velocity = np.zeros(2, dtype=np.float64)
        self._pan_velocity = np.zeros(3, dtype=np.float64)
        self._pressed_keys: set[int] = set()
        self._last_animation_time = time.perf_counter()
        self._frame_times: deque[float] = deque(maxlen=180)
        self._fps = 0.0

        self._functions: QOpenGLFunctions | None = None
        self._program: QOpenGLShaderProgram | None = None
        self._sobel_program: QOpenGLShaderProgram | None = None
        self._background_program: QOpenGLShaderProgram | None = None
        self._point_buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._grid_buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._measurement_line_buffer = QOpenGLBuffer(
            QOpenGLBuffer.Type.VertexBuffer
        )
        self._measurement_marker_buffer = QOpenGLBuffer(
            QOpenGLBuffer.Type.VertexBuffer
        )
        self._screen_buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._sobel_fbo: QOpenGLFramebufferObject | None = None
        self._point_count = 0
        self._grid_vertex_count = 0
        self._measurement_line_count = 0
        self._measurement_marker_count = 0
        self._display_indices = np.empty(0, dtype=np.int64)
        self._points_dirty = True
        self._grid_dirty = True
        self._measurement_dirty = True
        self._gl_ready = False
        self._gl_error = ''

        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(16)
        self._animation_timer.timeout.connect(self._animate_frame)
        self._animation_timer.start()

    def set_cloud(self, cloud: PointCloudData) -> None:
        self.cloud = cloud
        self.visible_classes = {int(value) for value in cloud.unique_classes}
        self.class_colors = {
            class_id: QColor(class_style_for(class_id).hex_color)
            for class_id in self.visible_classes
        }
        self.class_sizes = {
            class_id: 0.30 if class_id == 2 else 0.70 if class_id == 6 else 0.15
            for class_id in self.visible_classes
        }
        self.class_multipliers = {
            class_id: 1.0 for class_id in self.visible_classes
        }
        extents = cloud.bounds_max - cloud.bounds_min
        self._scene_radius = max(float(np.linalg.norm(extents)) / 2.0, 1.0)
        self.target = np.zeros(3, dtype=np.float64)
        self.reset_camera()
        self.clip_min = float(cloud.bounds_min[2])
        self.clip_max = float(cloud.bounds_max[2])
        self._points_dirty = True
        self._grid_dirty = True
        self._measurement_dirty = True
        self.clear_tools()
        self.update()

    def reset_camera(self) -> None:
        self.yaw = math.radians(-42.0)
        self.pitch = math.radians(34.0)
        self.distance = max(self._scene_radius * 2.3, 2.0)
        self.target = np.zeros(3, dtype=np.float64)
        self._angular_velocity.fill(0.0)
        self._pan_velocity.fill(0.0)
        self._pressed_keys.clear()
        self._emit_camera_position()
        self.update()

    def set_view(self, view: str) -> None:
        presets = {
            'top': (math.radians(-90), math.radians(89.0)),
            'bottom': (math.radians(90), math.radians(-89.0)),
            'front': (math.radians(-90), 0.0),
            'back': (math.radians(90), 0.0),
            'left': (math.radians(180), 0.0),
            'right': (0.0, 0.0),
        }
        if view in presets:
            eye = self._camera_position()
            self.yaw, self.pitch = presets[view]
            if self.navigation_mode == 'flight':
                self.target = eye - self._eye_direction() * self.distance
            self._emit_camera_position()
            self.update()

    def set_projection(self, perspective: bool) -> None:
        self.perspective = perspective
        self.update()

    def set_field_of_view(self, degrees: float) -> None:
        self.field_of_view = float(np.clip(degrees, 20.0, 120.0))
        self.update()

    def set_navigation_mode(self, mode: str) -> None:
        if mode not in {'orbit', 'flight'}:
            raise ValueError(f'Неизвестный режим камеры: {mode}')
        self.navigation_mode = mode
        self._pressed_keys.clear()
        self._angular_velocity.fill(0.0)
        self._pan_velocity.fill(0.0)
        self._last_animation_time = time.perf_counter()
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        self._emit_camera_position()
        self.update()

    def set_flight_speed(self, speed: float) -> None:
        self.flight_speed = float(np.clip(speed, 0.1, 100_000.0))

    def set_grid_visible(self, visible: bool) -> None:
        self.show_grid = bool(visible)
        self.update()

    def set_gradient_background(self, enabled: bool) -> None:
        self.gradient_background = bool(enabled)
        self.update()

    def set_point_size(self, size: float) -> None:
        self.point_size = max(1.0, float(size))
        self.update()

    def set_round_points(self, enabled: bool) -> None:
        self.round_points = enabled
        self.update()

    def set_edge_detection(self, enabled: bool) -> None:
        self.edge_detection = enabled
        self.update()

    def set_size_attenuation(self, enabled: bool) -> None:
        self.size_attenuation = enabled
        self.update()

    def set_gradient(self, name: str) -> None:
        if name not in GRADIENTS:
            raise ValueError(f'Неизвестный градиент: {name}')
        self.gradient_name = name
        self._points_dirty = True
        self.update()

    def set_color_mode(self, mode: str) -> None:
        if mode not in {'classification', 'height', 'rgb'}:
            raise ValueError(f'Неизвестный режим цвета: {mode}')
        self.color_mode = mode
        self._points_dirty = True
        self.update()

    def set_class_visible(self, class_id: int, visible: bool) -> None:
        if visible:
            self.visible_classes.add(class_id)
        else:
            self.visible_classes.discard(class_id)
        self._points_dirty = True
        self.update()

    def set_class_color(self, class_id: int, color: QColor) -> None:
        self.class_colors[class_id] = QColor(color)
        self._points_dirty = True
        self.update()

    def set_class_size(self, class_id: int, scale: float) -> None:
        self.class_sizes[class_id] = max(0.0, float(scale))
        self._points_dirty = True
        self.update()

    def set_class_multiplier(self, class_id: int, multiplier: float) -> None:
        self.class_multipliers[class_id] = max(0.1, float(multiplier))
        self._points_dirty = True
        self.update()

    def set_measurement_colors(self, marker: QColor, line: QColor) -> None:
        self.marker_color = QColor(marker)
        self.line_color = QColor(line)
        self._measurement_dirty = True
        self.update()

    def set_profile_width(self, width: float) -> None:
        self.profile_width = max(0.01, float(width))
        if self._profile_segment is not None and self.profile_clip_enabled:
            self._points_dirty = True
        self._measurement_dirty = True
        self.update()

    def set_profile_clip_mode(self, mode: str) -> None:
        if mode not in {'inside', 'outside', 'none'}:
            raise ValueError(f'Неизвестный режим профильного среза: {mode}')
        self.profile_clip_mode = mode
        self._points_dirty = True
        self.update()

    def set_profile_clipping_enabled(self, enabled: bool) -> None:
        self.profile_clip_enabled = bool(enabled)
        self._points_dirty = True
        self.update()

    def set_profile_helpers_visible(self, visible: bool) -> None:
        self.profile_helpers_visible = bool(visible)
        self._measurement_dirty = True
        self.update()

    def set_profile_segment(self, start: np.ndarray, end: np.ndarray) -> None:
        if self.cloud is None:
            return
        start_value = np.asarray(start, dtype=np.float64)
        end_value = np.asarray(end, dtype=np.float64)
        self._profile_segment = (start_value.copy(), end_value.copy())
        self._profile_points = [
            start_value - self.cloud.center,
            end_value - self.cloud.center,
        ]
        self._points_dirty = True
        self._measurement_dirty = True
        self.update()

    def fit_profile_view(self, start: np.ndarray, end: np.ndarray) -> None:
        if self.cloud is None:
            return
        start_centered = np.asarray(start, dtype=np.float64) - self.cloud.center
        end_centered = np.asarray(end, dtype=np.float64) - self.cloud.center
        direction = end_centered - start_centered
        length = max(float(np.linalg.norm(direction[:2])), 1.0)
        self.target = (start_centered + end_centered) / 2.0
        self.yaw = math.atan2(direction[1], direction[0]) + math.pi / 2.0
        self.pitch = 0.0
        z_span = max(float(self.cloud.bounds_max[2] - self.cloud.bounds_min[2]), 1.0)
        self.distance = max(length * 0.7, z_span * 1.4, 2.0)
        self.perspective = False
        self._emit_camera_position()
        self.update()

    def set_clip(self, enabled: bool, minimum: float, maximum: float) -> None:
        self.clip_enabled = enabled
        self.clip_min = min(minimum, maximum)
        self.clip_max = max(minimum, maximum)
        self._points_dirty = True
        self.update()

    def set_tool_mode(self, mode: str) -> None:
        if mode not in {'camera', 'info', 'ruler', 'profile'}:
            raise ValueError(f'Неизвестный инструмент: {mode}')
        self.tool_mode = mode
        cursors = {
            'camera': Qt.CursorShape.OpenHandCursor,
            'info': Qt.CursorShape.CrossCursor,
            'ruler': Qt.CursorShape.CrossCursor,
            'profile': Qt.CursorShape.CrossCursor,
        }
        self.setCursor(cursors[mode])

    def clear_tools(self) -> None:
        self._ruler_paths = [[]]
        self._profile_points.clear()
        self._profile_segment = None
        self._info_points.clear()
        self._hover_point = None
        self._hover_label = ''
        self._points_dirty = True
        self._measurement_dirty = True
        self.measurement_changed.emit('Измерения очищены')
        self.update()

    def initializeGL(self) -> None:
        self._functions = QOpenGLFunctions(self.context())
        self._functions.initializeOpenGLFunctions()
        self._functions.glEnable(GL_DEPTH_TEST)
        self._functions.glEnable(GL_BLEND)
        self._functions.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._functions.glEnable(GL_PROGRAM_POINT_SIZE)
        self._functions.glClearColor(0.035, 0.055, 0.09, 1.0)

        program = QOpenGLShaderProgram(self)
        if not program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex,
            VERTEX_SHADER,
        ):
            self._set_gl_error(program.log())
            return
        if not program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment,
            FRAGMENT_SHADER,
        ):
            self._set_gl_error(program.log())
            return
        program.bindAttributeLocation('position', 0)
        program.bindAttributeLocation('color', 1)
        program.bindAttributeLocation('pointScale', 2)
        if not program.link():
            self._set_gl_error(program.log())
            return

        self._program = program
        sobel_program = QOpenGLShaderProgram(self)
        if not sobel_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex,
            SCREEN_VERTEX_SHADER,
        ):
            self._set_gl_error(sobel_program.log())
            return
        if not sobel_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment,
            SOBEL_FRAGMENT_SHADER,
        ):
            self._set_gl_error(sobel_program.log())
            return
        sobel_program.bindAttributeLocation('position', 0)
        if not sobel_program.link():
            self._set_gl_error(sobel_program.log())
            return
        self._sobel_program = sobel_program

        background_program = QOpenGLShaderProgram(self)
        if not background_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex,
            SCREEN_VERTEX_SHADER,
        ):
            self._set_gl_error(background_program.log())
            return
        if not background_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment,
            BACKGROUND_FRAGMENT_SHADER,
        ):
            self._set_gl_error(background_program.log())
            return
        background_program.bindAttributeLocation('position', 0)
        if not background_program.link():
            self._set_gl_error(background_program.log())
            return
        self._background_program = background_program

        self._point_buffer.create()
        self._point_buffer.setUsagePattern(QOpenGLBuffer.UsagePattern.DynamicDraw)
        self._grid_buffer.create()
        self._grid_buffer.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
        self._measurement_line_buffer.create()
        self._measurement_line_buffer.setUsagePattern(
            QOpenGLBuffer.UsagePattern.DynamicDraw
        )
        self._measurement_marker_buffer.create()
        self._measurement_marker_buffer.setUsagePattern(
            QOpenGLBuffer.UsagePattern.DynamicDraw
        )
        self._screen_buffer.create()
        self._screen_buffer.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
        quad = np.asarray(
            ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)),
            dtype=np.float32,
        )
        if self._screen_buffer.bind():
            payload = np.ascontiguousarray(quad).tobytes()
            self._screen_buffer.allocate(payload, len(payload))
            self._screen_buffer.release()
        self._gl_ready = True

    def _set_gl_error(self, message: str) -> None:
        self._gl_error = message or 'Не удалось инициализировать OpenGL.'
        self.render_error.emit(self._gl_error)

    def resizeGL(self, width: int, height: int) -> None:
        self._sobel_fbo = None
        if self._functions:
            self._functions.glViewport(0, 0, width, max(1, height))

    def paintGL(self) -> None:
        if not self._functions:
            return
        if not self._gl_ready or self._program is None:
            self._functions.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._draw_overlay()
            return

        if self._grid_dirty:
            self._upload_grid()
        if self._points_dirty:
            self._upload_points()
        if self._measurement_dirty:
            self._upload_measurements()

        if self.edge_detection and self._sobel_program is not None:
            self._render_with_sobel()
        else:
            self._functions.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._render_scene()

        now = time.perf_counter()
        self._frame_times.append(now)
        while self._frame_times and now - self._frame_times[0] > 1.0:
            self._frame_times.popleft()
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            self._fps = (len(self._frame_times) - 1) / max(elapsed, 1e-9)
        self._draw_overlay()

    def _render_scene(self) -> None:
        if self._program is None:
            return
        if self.gradient_background:
            self._render_background()
        projection, view = self._camera_matrices()
        matrix = projection * view
        self._program.bind()
        self._program.setUniformValue('mvp', matrix)
        self._program.setUniformValue('roundPoints', self.round_points)
        self._program.setUniformValue1f(
            self._program.uniformLocation('perspectiveScale'),
            perspective_point_scale(
                self.height(),
                self.devicePixelRatioF(),
                self.field_of_view,
            ),
        )
        self._program.setUniformValue(
            'sizeAttenuation',
            self.size_attenuation and self.perspective,
        )

        if self.show_grid and self._grid_vertex_count:
            self._program.setUniformValue('pointPass', False)
            self._program.setUniformValue('overrideColor', False)
            self._program.setUniformValue1f(
                self._program.uniformLocation('pointSize'),
                1.0,
            )
            self._draw_buffer(self._grid_buffer, GL_LINES, self._grid_vertex_count)

        if self._point_count:
            self._program.setUniformValue('pointPass', True)
            self._program.setUniformValue('overrideColor', False)
            point_size = self.point_size
            if not (self.size_attenuation and self.perspective):
                point_size *= max(1.0, self.devicePixelRatioF())
            self._program.setUniformValue1f(
                self._program.uniformLocation('pointSize'),
                point_size,
            )
            self._draw_buffer(self._point_buffer, GL_POINTS, self._point_count)

        if self._measurement_line_count:
            self._program.setUniformValue('pointPass', False)
            self._program.setUniformValue('overrideColor', False)
            self._program.setUniformValue1f(
                self._program.uniformLocation('pointSize'),
                1.0,
            )
            self._functions.glLineWidth(
                max(1.0, 2.2 * self.devicePixelRatioF())
            )
            self._draw_buffer(
                self._measurement_line_buffer,
                GL_LINES,
                self._measurement_line_count,
            )
            self._functions.glLineWidth(1.0)

        if self._measurement_marker_count:
            self._functions.glDepthFunc(GL_LEQUAL)
            self._program.setUniformValue('pointPass', True)
            self._program.setUniformValue('roundPoints', True)
            self._program.setUniformValue('sizeAttenuation', True)
            # Маркеры сохраняют прежний визуальный размер, но остаются
            # глубинными 3D-объектами и меняются при приближении камеры.
            self._program.setUniformValue1f(
                self._program.uniformLocation('perspectiveScale'),
                120.0,
            )
            self._program.setUniformValue('overrideColor', False)
            self._program.setUniformValue1f(
                self._program.uniformLocation('pointSize'),
                10.0 * max(1.0, self.devicePixelRatioF()),
            )
            self._draw_buffer(
                self._measurement_marker_buffer,
                GL_POINTS,
                self._measurement_marker_count,
            )
            self._functions.glDepthFunc(GL_LESS)

        self._program.release()

    def _render_background(self) -> None:
        if (
            not self._functions
            or self._background_program is None
            or not self._screen_buffer.bind()
        ):
            return
        self._functions.glDisable(GL_DEPTH_TEST)
        self._background_program.bind()
        self._background_program.setUniformValue(
            'topColor',
            QVector3D(0.075, 0.145, 0.255),
        )
        self._background_program.setUniformValue(
            'bottomColor',
            QVector3D(0.020, 0.030, 0.060),
        )
        self._background_program.enableAttributeArray(0)
        self._background_program.setAttributeBuffer(0, GL_FLOAT, 0, 2, 0)
        self._functions.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        self._background_program.disableAttributeArray(0)
        self._screen_buffer.release()
        self._background_program.release()
        self._functions.glEnable(GL_DEPTH_TEST)

    def _render_with_sobel(self) -> None:
        if not self._functions or self._sobel_program is None:
            return
        pixel_ratio = max(1.0, self.devicePixelRatioF())
        pixel_width = max(1, round(self.width() * pixel_ratio))
        pixel_height = max(1, round(self.height() * pixel_ratio))
        if (
            self._sobel_fbo is None
            or self._sobel_fbo.width() != pixel_width
            or self._sobel_fbo.height() != pixel_height
        ):
            frame_format = QOpenGLFramebufferObjectFormat()
            frame_format.setAttachment(
                QOpenGLFramebufferObject.Attachment.CombinedDepthStencil
            )
            self._sobel_fbo = QOpenGLFramebufferObject(
                pixel_width,
                pixel_height,
                frame_format,
            )

        if not self._sobel_fbo.isValid() or not self._sobel_fbo.bind():
            self._functions.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._render_scene()
            return

        self._functions.glViewport(0, 0, pixel_width, pixel_height)
        self._functions.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._render_scene()
        self._sobel_fbo.release()

        self._functions.glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        self._functions.glViewport(0, 0, pixel_width, pixel_height)
        self._functions.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._functions.glDisable(GL_DEPTH_TEST)
        self._sobel_program.bind()
        self._sobel_program.setUniformValue('sceneTexture', 0)
        self._sobel_program.setUniformValue(
            'texelSize',
            QVector2D(1.0 / pixel_width, 1.0 / pixel_height),
        )
        self._functions.glActiveTexture(GL_TEXTURE0)
        self._functions.glBindTexture(GL_TEXTURE_2D, self._sobel_fbo.texture())
        if self._screen_buffer.bind():
            self._sobel_program.enableAttributeArray(0)
            self._sobel_program.setAttributeBuffer(0, GL_FLOAT, 0, 2, 0)
            self._functions.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            self._sobel_program.disableAttributeArray(0)
            self._screen_buffer.release()
        self._functions.glBindTexture(GL_TEXTURE_2D, 0)
        self._sobel_program.release()
        self._functions.glEnable(GL_DEPTH_TEST)

    def _draw_buffer(
        self,
        buffer: QOpenGLBuffer,
        primitive: int,
        count: int,
    ) -> None:
        if not self._program or not self._functions or not buffer.bind():
            return
        stride = 7 * np.dtype(np.float32).itemsize
        self._program.enableAttributeArray(0)
        self._program.enableAttributeArray(1)
        self._program.enableAttributeArray(2)
        self._program.setAttributeBuffer(0, GL_FLOAT, 0, 3, stride)
        self._program.setAttributeBuffer(
            1,
            GL_FLOAT,
            3 * np.dtype(np.float32).itemsize,
            3,
            stride,
        )
        self._program.setAttributeBuffer(
            2,
            GL_FLOAT,
            6 * np.dtype(np.float32).itemsize,
            1,
            stride,
        )
        self._functions.glDrawArrays(primitive, 0, count)
        self._program.disableAttributeArray(0)
        self._program.disableAttributeArray(1)
        self._program.disableAttributeArray(2)
        buffer.release()

    def _upload_points(self) -> None:
        self._points_dirty = False
        if self.cloud is None:
            self._point_count = 0
            self._display_indices = np.empty(0, dtype=np.int64)
            return

        mask = np.isin(self.cloud.classes, list(self.visible_classes))
        if self.clip_enabled:
            original_z = self.cloud.points[:, 2] + self.cloud.center[2]
            mask &= (original_z >= self.clip_min) & (original_z <= self.clip_max)
        if (
            self.profile_clip_enabled
            and self._profile_segment is not None
            and self.profile_clip_mode != 'none'
        ):
            start, end = self._profile_segment
            original_xy = self.cloud.original_points[:, :2]
            direction = end[:2] - start[:2]
            length = float(np.linalg.norm(direction))
            if length > 1e-9:
                unit = direction / length
                relative = original_xy - start[:2]
                along = relative @ unit
                perpendicular = np.abs(
                    relative[:, 0] * unit[1] - relative[:, 1] * unit[0]
                )
                corridor = (
                    (along >= 0.0)
                    & (along <= length)
                    & (perpendicular <= self.profile_width / 2.0)
                )
                mask &= corridor if self.profile_clip_mode == 'inside' else ~corridor
        self._display_indices = np.flatnonzero(mask)
        points = self.cloud.points[self._display_indices]
        colors = self._point_colors(self._display_indices)
        classes = self.cloud.classes[self._display_indices]
        scales = class_point_scales(
            classes,
            self.class_sizes,
            self.class_multipliers,
        )
        interleaved = np.empty((len(points), 7), dtype=np.float32)
        interleaved[:, :3] = points
        interleaved[:, 3:6] = colors
        interleaved[:, 6] = scales
        payload = np.ascontiguousarray(interleaved).tobytes()

        if self._point_buffer.bind():
            self._point_buffer.allocate(payload, len(payload))
            self._point_buffer.release()
        self._point_count = len(points)

    def _upload_measurements(self) -> None:
        self._measurement_dirty = False
        marker_rgb = (
            self.marker_color.redF(),
            self.marker_color.greenF(),
            self.marker_color.blueF(),
        )
        line_rgb = (
            self.line_color.redF(),
            self.line_color.greenF(),
            self.line_color.blueF(),
        )
        helper_rgb = (0.38, 0.65, 0.98)
        line_vertices: list[tuple[float, ...]] = []
        marker_vertices: list[tuple[float, ...]] = []

        def vertex(point: np.ndarray, color: tuple[float, float, float], scale: float = 1.0):
            return (
                float(point[0]),
                float(point[1]),
                float(point[2]),
                *color,
                scale,
            )

        for path in self._ruler_paths:
            marker_vertices.extend(vertex(point, marker_rgb, 1.0) for point in path)
            for index in range(1, len(path)):
                line_vertices.extend(
                    (
                        vertex(path[index - 1], line_rgb),
                        vertex(path[index], line_rgb),
                    )
                )

        marker_vertices.extend(
            vertex(point, line_rgb, 1.0) for point in self._profile_points
        )
        if len(self._profile_points) >= 2:
            line_vertices.extend(
                (
                    vertex(self._profile_points[0], line_rgb),
                    vertex(self._profile_points[1], line_rgb),
                )
            )

        marker_vertices.extend(
            vertex(point, marker_rgb, 1.15) for point, _label in self._info_points
        )
        if self._hover_point is not None:
            marker_vertices.append(vertex(self._hover_point, (0.22, 0.40, 0.96), 1.3))

        helper_corners = self._profile_helper_corners()
        if self.profile_helpers_visible and helper_corners is not None:
            for start_index, end_index in ((0, 1), (1, 2), (2, 3), (3, 0)):
                line_vertices.extend(
                    (
                        vertex(helper_corners[start_index], helper_rgb),
                        vertex(helper_corners[end_index], helper_rgb),
                    )
                )

        self._measurement_line_count = self._allocate_vertices(
            self._measurement_line_buffer,
            line_vertices,
        )
        self._measurement_marker_count = self._allocate_vertices(
            self._measurement_marker_buffer,
            marker_vertices,
        )

    @staticmethod
    def _allocate_vertices(
        buffer: QOpenGLBuffer,
        vertices: list[tuple[float, ...]],
    ) -> int:
        array = np.asarray(vertices, dtype=np.float32).reshape(-1, 7)
        payload = np.ascontiguousarray(array).tobytes()
        if buffer.bind():
            buffer.allocate(payload, len(payload))
            buffer.release()
        return len(array)

    def _point_colors(self, indices: np.ndarray) -> np.ndarray:
        assert self.cloud is not None
        if self.color_mode == 'rgb' and self.cloud.rgb is not None:
            return self.cloud.rgb[indices].astype(np.float32, copy=False)
        if self.color_mode == 'height':
            z_values = self.cloud.points[indices, 2]
            return self._height_colors(z_values)

        colors = np.empty((len(indices), 3), dtype=np.float32)
        classes = self.cloud.classes[indices]
        for class_id in np.unique(classes):
            color = self.class_colors.get(
                int(class_id),
                QColor(class_style_for(int(class_id)).hex_color),
            )
            colors[classes == class_id] = (
                color.redF(),
                color.greenF(),
                color.blueF(),
            )
        return colors

    def _height_colors(self, z_values: np.ndarray) -> np.ndarray:
        if not len(z_values):
            return np.empty((0, 3), dtype=np.float32)
        minimum = float(z_values.min())
        maximum = float(z_values.max())
        normalized = (z_values - minimum) / max(maximum - minimum, 1e-9)
        return interpolate_gradient(self.gradient_name, normalized)

    def _upload_grid(self) -> None:
        self._grid_dirty = False
        vertices = self._grid_vertices()
        payload = np.ascontiguousarray(vertices).tobytes()
        if self._grid_buffer.bind():
            self._grid_buffer.allocate(payload, len(payload))
            self._grid_buffer.release()
        self._grid_vertex_count = len(vertices)

    def _grid_vertices(self) -> np.ndarray:
        radius = max(self._scene_radius, 1.0)
        step = self._nice_step(radius * 2.0 / 10.0)
        limit = math.ceil(radius / step) * step
        floor_z = (
            float(self.cloud.bounds_min[2] - self.cloud.center[2])
            if self.cloud is not None
            else 0.0
        )
        vertices: list[tuple[float, float, float, float, float, float, float]] = []
        line_count = round(limit * 2.0 / step)
        grid_color = (0.20, 0.28, 0.39)
        for index in range(line_count + 1):
            value = -limit + index * step
            vertices.extend(
                (
                    (value, -limit, floor_z, *grid_color, 1.0),
                    (value, limit, floor_z, *grid_color, 1.0),
                    (-limit, value, floor_z, *grid_color, 1.0),
                    (limit, value, floor_z, *grid_color, 1.0),
                )
            )

        axis_length = limit * 0.42
        vertices.extend(
            (
                (0, 0, floor_z, 0.96, 0.30, 0.28, 1.0),
                (axis_length, 0, floor_z, 0.96, 0.30, 0.28, 1.0),
                (0, 0, floor_z, 0.28, 0.78, 0.46, 1.0),
                (0, axis_length, floor_z, 0.28, 0.78, 0.46, 1.0),
                (0, 0, floor_z, 0.32, 0.58, 0.98, 1.0),
                (0, 0, floor_z + axis_length, 0.32, 0.58, 0.98, 1.0),
            )
        )
        return np.asarray(vertices, dtype=np.float32)

    @staticmethod
    def _nice_step(value: float) -> float:
        exponent = math.floor(math.log10(max(value, 1e-9)))
        fraction = value / (10**exponent)
        if fraction <= 1:
            nice_fraction = 1
        elif fraction <= 2:
            nice_fraction = 2
        elif fraction <= 5:
            nice_fraction = 5
        else:
            nice_fraction = 10
        return nice_fraction * (10**exponent)

    def _eye_direction(self) -> np.ndarray:
        cos_pitch = math.cos(self.pitch)
        return np.asarray(
            [
                cos_pitch * math.cos(self.yaw),
                cos_pitch * math.sin(self.yaw),
                math.sin(self.pitch),
            ],
            dtype=np.float64,
        )

    def _camera_position(self) -> np.ndarray:
        return self.target + self._eye_direction() * self.distance

    def _camera_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eye = self._camera_position()
        forward = self.target - eye
        forward /= max(float(np.linalg.norm(forward)), 1e-9)
        world_up = np.asarray([0.0, 0.0, 1.0])
        if abs(float(forward @ world_up)) > 0.98:
            world_up = np.asarray([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= max(float(np.linalg.norm(right)), 1e-9)
        up = np.cross(right, forward)
        up /= max(float(np.linalg.norm(up)), 1e-9)
        return eye, forward, right, up

    def _camera_matrices(self) -> tuple[QMatrix4x4, QMatrix4x4]:
        width = max(1, self.width())
        height = max(1, self.height())
        aspect = width / height
        eye, _forward, _right, up = self._camera_basis()
        projection = QMatrix4x4()
        near = max(0.01, self._scene_radius / 10_000.0)
        far = max(
            100.0,
            self.distance + self._scene_radius * 20.0,
            float(np.linalg.norm(eye)) + self._scene_radius * 20.0,
        )
        if self.perspective:
            projection.perspective(self.field_of_view, aspect, near, far)
        else:
            span = max(self.distance * 0.56, self._scene_radius * 0.25)
            projection.ortho(-span * aspect, span * aspect, -span, span, -far, far)

        view = QMatrix4x4()
        view.lookAt(
            QVector3D(*eye),
            QVector3D(*self.target),
            QVector3D(*up),
        )
        return projection, view

    def _project_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not len(points):
            return np.empty((0, 2)), np.empty(0, dtype=bool)
        eye, forward, right, up = self._camera_basis()
        relative = points.astype(np.float64, copy=False) - eye
        depth = relative @ forward
        camera_x = relative @ right
        camera_y = relative @ up
        aspect = max(1, self.width()) / max(1, self.height())

        if self.perspective:
            tangent = math.tan(math.radians(self.field_of_view) / 2.0)
            safe_depth = np.maximum(depth, 1e-9)
            ndc_x = camera_x / (safe_depth * tangent * aspect)
            ndc_y = camera_y / (safe_depth * tangent)
            visible = depth > 0.0
        else:
            span = max(self.distance * 0.56, self._scene_radius * 0.25)
            ndc_x = camera_x / (span * aspect)
            ndc_y = camera_y / span
            visible = np.ones(len(points), dtype=bool)

        visible &= (np.abs(ndc_x) <= 1.05) & (np.abs(ndc_y) <= 1.05)
        screen = np.column_stack(
            (
                (ndc_x + 1.0) * 0.5 * self.width(),
                (1.0 - ndc_y) * 0.5 * self.height(),
            )
        )
        return screen, visible

    def _pick_point(self, position: QPointF) -> int | None:
        if self.cloud is None or not len(self._display_indices):
            return None
        points = self.cloud.points[self._display_indices]
        screen, visible = self._project_points(points)
        delta_x = screen[:, 0] - position.x()
        delta_y = screen[:, 1] - position.y()
        distance_squared = delta_x * delta_x + delta_y * delta_y
        distance_squared[~visible] = np.inf
        local_index = int(np.argmin(distance_squared))
        if float(distance_squared[local_index]) > 14.0**2:
            return None
        return int(self._display_indices[local_index])

    def _handle_pick(self, position: QPointF, start_new_path: bool = False) -> None:
        if self.cloud is None:
            return
        index = self._pick_point(position)
        if index is None:
            self.measurement_changed.emit('Рядом с курсором нет видимой точки')
            return

        centered = self.cloud.points[index].astype(np.float64)
        original = centered + self.cloud.center
        class_id = int(self.cloud.classes[index])
        details = {
            'index': index,
            'position': original.copy(),
            'class_id': class_id,
            'class_title': class_style_for(class_id).title,
        }
        self.point_picked.emit(details)

        if self.tool_mode == 'info':
            label = (
                f'X {original[0]:.3f} · Y {original[1]:.3f} · Z {original[2]:.3f}\n'
                f'{details["class_title"]} (класс {class_id})'
            )
            self._info_points.append((centered.copy(), label))
            self.measurement_changed.emit(
                f'X {original[0]:.3f}  ·  Y {original[1]:.3f}  ·  '
                f'Z {original[2]:.3f}  ·  {details["class_title"]}'
            )
        elif self.tool_mode == 'ruler':
            if start_new_path or not self._ruler_paths:
                self._ruler_paths.append([])
            path = self._ruler_paths[-1]
            path.append(centered.copy())
            if len(path) >= 2:
                segment = float(np.linalg.norm(path[-1] - path[-2]))
                total = sum(
                    float(np.linalg.norm(path[i] - path[i - 1]))
                    for i in range(1, len(path))
                )
                self.measurement_changed.emit(
                    f'Отрезок: {segment:.3f} м · длина линии: {total:.3f} м'
                )
            else:
                self.measurement_changed.emit(
                    'Выберите следующую точку · Shift+ЛКМ — новая линия'
                )
        elif self.tool_mode == 'profile':
            if len(self._profile_points) == 2:
                self._profile_points.clear()
            self._profile_points.append(centered.copy())
            if len(self._profile_points) == 2:
                self._profile_segment = (
                    self._profile_points[0] + self.cloud.center,
                    self._profile_points[1] + self.cloud.center,
                )
                self.profile_ready.emit(
                    *self._profile_segment,
                )
                if self.profile_clip_enabled:
                    self._points_dirty = True
                self.measurement_changed.emit('Профиль построен по выбранному коридору')
            else:
                self.measurement_changed.emit('Выберите конечную точку профиля')
        self._measurement_dirty = True
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._last_mouse_position = event.position().toPoint()
        self._drag_button = event.button()
        if event.button() == Qt.MouseButton.LeftButton and self.tool_mode != 'camera':
            self._handle_pick(
                event.position(),
                bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier),
            )
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        current = event.position().toPoint()
        delta = current - self._last_mouse_position
        self._last_mouse_position = current
        buttons = event.buttons()

        if buttons & Qt.MouseButton.LeftButton and self.tool_mode == 'camera':
            eye = self._camera_position()
            yaw_delta = -delta.x() * 0.008
            pitch_delta = delta.y() * 0.008
            self.yaw += yaw_delta
            self.pitch = float(
                np.clip(self.pitch + pitch_delta, math.radians(-89), math.radians(89))
            )
            if self.navigation_mode == 'flight':
                self.target = eye - self._eye_direction() * self.distance
                self._angular_velocity.fill(0.0)
            else:
                self._angular_velocity[:] = (yaw_delta, pitch_delta)
            self._emit_camera_position()
            self.update()
        elif buttons & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
            _eye, _forward, right, up = self._camera_basis()
            scale = self.distance / max(220.0, self.height())
            movement = (-right * delta.x() + up * delta.y()) * scale
            self.target += movement
            self._pan_velocity = movement
            self._emit_camera_position()
            self.update()
        elif (
            self.tool_mode != 'camera'
            or event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            now = time.perf_counter()
            if now - self._last_hover_pick >= 0.06:
                self._last_hover_pick = now
                self._update_hover(event.position())
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_button = Qt.MouseButton.NoButton
        if self.tool_mode == 'camera':
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120.0
        if self.navigation_mode == 'flight':
            _eye, forward, _right, _up = self._camera_basis()
            self.target += forward * steps * self.flight_speed * 0.25
        else:
            self.distance *= math.pow(0.86, steps)
            self.distance = float(
                np.clip(
                    self.distance,
                    self._scene_radius * 0.015,
                    self._scene_radius * 80.0,
                )
            )
        self._emit_camera_position()
        self.update()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Home:
            self.reset_camera()
            event.accept()
            return
        if self.navigation_mode == 'flight' and key in FLIGHT_KEYS:
            self._pressed_keys.add(key)
            event.accept()
            return
        if key in {Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_W}:
            self.distance = max(self._scene_radius * 0.015, self.distance * 0.88)
        elif key in {Qt.Key.Key_Minus, Qt.Key.Key_Underscore, Qt.Key.Key_S}:
            self.distance = min(self._scene_radius * 80.0, self.distance / 0.88)
        elif key in {
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        }:
            _eye, _forward, right, up = self._camera_basis()
            step = max(self.distance * 0.035, self._scene_radius * 0.002)
            if key == Qt.Key.Key_Left:
                self.target -= right * step
            elif key == Qt.Key.Key_Right:
                self.target += right * step
            elif key == Qt.Key.Key_Up:
                self.target += up * step
            else:
                self.target -= up * step
        else:
            super().keyPressEvent(event)
            return
        self._emit_camera_position()
        self.update()
        event.accept()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if self.navigation_mode == 'flight' and key in FLIGHT_KEYS:
            self._pressed_keys.discard(key)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        self._pressed_keys.clear()
        super().focusOutEvent(event)

    def _animate_frame(self) -> None:
        now = time.perf_counter()
        elapsed = min(max(now - self._last_animation_time, 0.0), 0.05)
        self._last_animation_time = now
        changed = False
        if self.navigation_mode == 'flight' and self.tool_mode == 'camera':
            _eye, forward, right, _up = self._camera_basis()
            movement = flight_displacement(
                forward,
                right,
                float(Qt.Key.Key_W in self._pressed_keys)
                - float(Qt.Key.Key_S in self._pressed_keys),
                float(Qt.Key.Key_D in self._pressed_keys)
                - float(Qt.Key.Key_A in self._pressed_keys),
                float(Qt.Key.Key_Space in self._pressed_keys)
                - float(Qt.Key.Key_Shift in self._pressed_keys),
                self.flight_speed,
                elapsed,
            )
            if float(np.linalg.norm(movement)) > 0.0:
                self.target += movement
                changed = True
        if self._drag_button == Qt.MouseButton.NoButton:
            if (
                self.navigation_mode == 'orbit'
                and float(np.linalg.norm(self._angular_velocity)) > 0.00008
            ):
                self.yaw += float(self._angular_velocity[0])
                self.pitch = float(
                    np.clip(
                        self.pitch + float(self._angular_velocity[1]),
                        math.radians(-89),
                        math.radians(89),
                    )
                )
                self._angular_velocity *= 0.90
                changed = True
            else:
                self._angular_velocity.fill(0.0)
            if float(np.linalg.norm(self._pan_velocity)) > self._scene_radius * 1e-6:
                self.target += self._pan_velocity
                self._pan_velocity *= 0.86
                changed = True
            else:
                self._pan_velocity.fill(0.0)
        if changed:
            self._emit_camera_position()
        if self.isVisible():
            self.update()

    def _update_hover(self, position: QPointF) -> None:
        if self.cloud is None:
            return
        index = self._pick_point(position)
        if index is None:
            self._hover_point = None
            self._hover_label = ''
        else:
            centered = self.cloud.points[index].astype(np.float64)
            original = centered + self.cloud.center
            class_id = int(self.cloud.classes[index])
            self._hover_point = centered
            self._hover_label = (
                f'{original[0]:.2f}, {original[1]:.2f}, {original[2]:.2f} · '
                f'{class_style_for(class_id).title}'
            )
        self._measurement_dirty = True
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_camera()
        event.accept()

    def _emit_camera_position(self) -> None:
        eye = self._camera_position()
        mode = 'полёт' if self.navigation_mode == 'flight' else 'орбита'
        self.camera_changed.emit(
            f'Камера ({mode}): X {eye[0]:.2f} · Y {eye[1]:.2f} · Z {eye[2]:.2f}'
        )

    def _draw_overlay(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._gl_error:
            painter.setPen(QColor('#FCA5A5'))
            painter.drawText(
                self.rect().adjusted(30, 30, -30, -30),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                f'Ошибка OpenGL\n{self._gl_error}',
            )
            painter.end()
            return

        if self.cloud is None:
            painter.setPen(QColor('#94A3B8'))
            font = painter.font()
            font.setPointSize(12)
            font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                'Выберите или перетащите файл облака точек',
            )
            painter.end()
            return

        self._draw_selection_overlay(painter)
        self._draw_axis_overlay(painter)
        self._draw_fps_overlay(painter)
        painter.end()

    def _draw_selection_overlay(self, painter: QPainter) -> None:
        for path in self._ruler_paths:
            if not path:
                continue
            projected, visible = self._project_points(np.asarray(path))
            for index, point in enumerate(projected):
                if index == 0 or not (visible[index] and visible[index - 1]):
                    continue
                start = QPointF(*projected[index - 1])
                end = QPointF(*point)
                distance = float(np.linalg.norm(path[index] - path[index - 1]))
                self._draw_label(
                    painter,
                    (start + end) / 2.0,
                    f'{distance:.3f} м',
                )

        if self.profile_helpers_visible:
            self._draw_profile_helpers(painter)

        for point, label in self._info_points:
            projected, visible = self._project_points(point.reshape(1, 3))
            if not visible[0]:
                continue
            screen_point = QPointF(*projected[0])
            self._draw_label(painter, screen_point + QPointF(12.0, -12.0), label)

        if self._hover_point is not None:
            projected, visible = self._project_points(self._hover_point.reshape(1, 3))
            if visible[0]:
                screen_point = QPointF(*projected[0])
                self._draw_label(
                    painter,
                    screen_point + QPointF(12.0, 18.0),
                    self._hover_label,
                )

    def _draw_profile_helpers(self, painter: QPainter) -> None:
        corners = self._profile_helper_corners()
        if corners is None:
            return
        projected, visible = self._project_points(corners)
        if visible.all():
            self._draw_label(
                painter,
                QPointF(*projected[0]),
                f'профиль {self.profile_width:.2f} м',
            )

    def _profile_helper_corners(self) -> np.ndarray | None:
        if self.cloud is None or self._profile_segment is None:
            return None
        start, end = self._profile_segment
        direction = end[:2] - start[:2]
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            return None
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64) / length
        offset = normal * self.profile_width / 2.0
        return np.asarray(
            [
                (*((start[:2] + offset) - self.cloud.center[:2]), start[2] - self.cloud.center[2]),
                (*((end[:2] + offset) - self.cloud.center[:2]), end[2] - self.cloud.center[2]),
                (*((end[:2] - offset) - self.cloud.center[:2]), end[2] - self.cloud.center[2]),
                (*((start[:2] - offset) - self.cloud.center[:2]), start[2] - self.cloud.center[2]),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _draw_label(painter: QPainter, anchor: QPointF, text: str) -> None:
        metrics = painter.fontMetrics()
        lines = text.splitlines() or ['']
        width = max(metrics.horizontalAdvance(line) for line in lines) + 18
        height = metrics.height() * len(lines) + 12
        rect = QRectF(anchor.x() - width / 2.0, anchor.y() - height - 8.0, width, height)
        painter.setPen(QPen(QColor(71, 85, 105, 210), 1.0))
        painter.setBrush(QColor(11, 18, 32, 225))
        painter.drawRoundedRect(rect, 6.0, 6.0)
        painter.setPen(QColor('#F8FAFC'))
        painter.drawText(
            rect.adjusted(9, 5, -9, -5),
            Qt.AlignmentFlag.AlignCenter,
            text,
        )

    def _draw_fps_overlay(self, painter: QPainter) -> None:
        rect = QRectF(self.width() - 91.0, 14.0, 76.0, 28.0)
        painter.setPen(QPen(QColor(71, 85, 105, 190), 1.0))
        painter.setBrush(QColor(11, 18, 32, 205))
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.setPen(QColor('#A7F3D0'))
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignCenter,
            f'{self._fps:0.0f} FPS',
        )

    def _draw_axis_overlay(self, painter: QPainter) -> None:
        origin = QPointF(58.0, self.height() - 58.0)
        _eye, _forward, right, up = self._camera_basis()
        axes = (
            (np.asarray([1.0, 0.0, 0.0]), QColor('#F87171'), 'X'),
            (np.asarray([0.0, 1.0, 0.0]), QColor('#4ADE80'), 'Y'),
            (np.asarray([0.0, 0.0, 1.0]), QColor('#60A5FA'), 'Z'),
        )
        painter.setBrush(QColor(11, 18, 32, 190))
        painter.setPen(QPen(QColor(71, 85, 105, 170), 1.0))
        painter.drawEllipse(origin, 36.0, 36.0)
        for axis, color, label in axes:
            x_component = float(axis @ right)
            y_component = float(axis @ up)
            end = origin + QPointF(x_component * 28.0, -y_component * 28.0)
            painter.setPen(QPen(color, 2.5))
            painter.drawLine(origin, end)
            painter.drawText(end + QPointF(3.0, -3.0), label)
