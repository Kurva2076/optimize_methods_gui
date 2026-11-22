# Источник вычислительного ядра ML Studio

В эту ветку перенесены Python-скрипты проекта команды тестирования:

- репозиторий: <https://github.com/eliotBenitez/pointnet-hpo-mlstudio>;
- исходный коммит: `ed33a7a61713b79a81a3bf93e8cb5144ca4ffe5a`;
- перенесённые файлы: `hpo_core/`, `hpo_pointnet.py`, `run_all.py`,
  `predict_scene.py`, `summary_report.py`, `mlstudio_engine_checks.py` и скрипты
  настройки окружения `run_all.*`.

Исходный `test_data_summary.py` переименован в `mlstudio_engine_checks.py`,
чтобы тяжёлые проверки PyTorch/Optuna не попадали в быстрый набор GUI-тестов
при обычном запуске `pytest`.

Rust/GPUI-интерфейс не включён в сборку. Его пользовательские сценарии
реализованы нативными экранами PySide6, чтобы сохранить единую бело-синюю
оболочку PointCloud и встроенный OpenGL-просмотрщик.

