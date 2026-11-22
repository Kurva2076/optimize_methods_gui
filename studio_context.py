"""Общий контекст новых экранов ML Studio."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Signal

from studio_data import is_project
from studio_processes import ProcessRegistry


class StudioContext(QObject):
    project_changed = Signal(object)
    recent_projects_changed = Signal()
    favorites_changed = Signal()
    theme_changed = Signal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings or QSettings('PointCloud', 'MLStudio')
        self.processes = ProcessRegistry(self)
        self.project_root: Path | None = None
        saved = self.settings.value('project_root', '', type=str)
        if saved and is_project(saved):
            self.project_root = Path(saved).resolve()
        else:
            application_root = (
                Path(sys.executable).resolve().parent
                if getattr(sys, 'frozen', False)
                else Path(__file__).resolve().parent
            )
            if is_project(application_root):
                self.project_root = application_root

    @property
    def recent_projects(self) -> list[Path]:
        return self._stored_projects('recent_projects')

    @property
    def favorite_projects(self) -> list[Path]:
        return self._stored_projects('favorite_projects')

    @property
    def theme(self) -> str:
        value = self.settings.value('theme', 'light', type=str)
        return value if value in {'light', 'dark'} else 'light'

    def _stored_projects(self, key: str) -> list[Path]:
        raw = self.settings.value(key, '[]', type=str)
        try:
            values = json.loads(raw)
        except (TypeError, ValueError):
            values = []
        projects: list[Path] = []
        for value in values if isinstance(values, list) else []:
            try:
                project = Path(value).expanduser().resolve()
            except (OSError, TypeError, ValueError):
                continue
            if is_project(project) and project not in projects:
                projects.append(project)
        return projects

    def _save_projects(self, key: str, projects: list[Path]) -> None:
        self.settings.setValue(key, json.dumps([str(item) for item in projects]))

    def set_project(self, path: str | Path) -> bool:
        root = Path(path).expanduser().resolve()
        if not is_project(root):
            return False
        self._restore_predict_script(root)
        self.project_root = root
        recent = [item for item in self.recent_projects if item != root]
        recent.insert(0, root)
        recent = recent[:10]
        self.settings.setValue('project_root', str(root))
        self._save_projects('recent_projects', recent)
        self.project_changed.emit(root)
        self.recent_projects_changed.emit()
        return True

    @staticmethod
    def _restore_predict_script(root: Path) -> None:
        """Восстановить совместимый predict_scene.py из сборки.

        Файл с маркером ``model=dict(`` считаем пользовательским и не
        перезаписываем. Отсутствующая или очевидно повреждённая копия
        заменяется атомарно через временный файл.
        """
        source = Path(__file__).resolve().parent / 'predict_scene.py'
        target = root / 'predict_scene.py'
        if not source.is_file() or source.resolve() == target.resolve():
            return
        try:
            if target.is_file() and 'model=dict(' in target.read_text(
                encoding='utf-8',
                errors='ignore',
            ):
                return
            temporary = target.with_suffix('.py.tmp')
            shutil.copy2(source, temporary)
            temporary.replace(target)
        except OSError:
            # Открытие read-only проекта не должно ронять GUI;
            # конкретную ошибку покажет форма теста при запуске.
            return

    def is_favorite(self, path: str | Path) -> bool:
        return Path(path).expanduser().resolve() in self.favorite_projects

    def set_favorite(self, path: str | Path, favorite: bool = True) -> bool:
        root = Path(path).expanduser().resolve()
        if not is_project(root):
            return False
        projects = [item for item in self.favorite_projects if item != root]
        if favorite:
            projects.insert(0, root)
        self._save_projects('favorite_projects', projects[:20])
        self.favorites_changed.emit()
        return True

    def toggle_favorite(self, path: str | Path) -> bool:
        root = Path(path).expanduser().resolve()
        favorite = not self.is_favorite(root)
        return self.set_favorite(root, favorite)

    def remove_recent(self, path: str | Path) -> None:
        root = Path(path).expanduser().resolve()
        projects = [item for item in self.recent_projects if item != root]
        self._save_projects('recent_projects', projects)
        self.recent_projects_changed.emit()

    def set_theme(self, theme: str) -> None:
        if theme not in {'light', 'dark'}:
            raise ValueError(f'Неизвестная тема: {theme}')
        if theme == self.theme:
            return
        self.settings.setValue('theme', theme)
        self.theme_changed.emit(theme)

    def toggle_theme(self) -> str:
        theme = 'dark' if self.theme == 'light' else 'light'
        self.set_theme(theme)
        return theme

    def python_executable(self) -> Path | None:
        """Python для ML-скриптов.

        В обычном запуске можно использовать текущий интерпретатор. В
        PyInstaller-сборке ``sys.executable`` указывает на GUI exe, а не на
        Python. В этом случае запуск разрешаем только через подготовленную
        ``.venv311`` — иначе exe запустил бы вторую копию самого себя.
        """
        if self.project_root is not None:
            if os.name == 'nt':
                candidate = self.project_root / '.venv311' / 'Scripts' / 'python.exe'
            else:
                candidate = self.project_root / '.venv311' / 'bin' / 'python'
            if candidate.is_file():
                return candidate
        if getattr(sys, 'frozen', False):
            return None
        return Path(sys.executable)

    def runs_directory(self) -> Path | None:
        return self.project_root / 'runs' if self.project_root is not None else None

