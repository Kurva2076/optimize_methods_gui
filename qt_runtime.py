"""Подготовка платформенных Qt-плагинов перед созданием QApplication."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


def clear_hidden_flags(root: Path) -> int:
    """Снять у каталога и его содержимого macOS-флаг UF_HIDDEN."""
    hidden_flag = getattr(stat, 'UF_HIDDEN', 0)
    if not hidden_flag or not root.exists():
        return 0

    changed = 0
    paths = [root]
    for directory, subdirectories, filenames in os.walk(root):
        current = Path(directory)
        paths.extend(current / name for name in subdirectories)
        paths.extend(current / name for name in filenames)

    for path in paths:
        try:
            flags = os.stat(path, follow_symlinks=False).st_flags
            if flags & hidden_flag:
                os.chflags(
                    path,
                    flags & ~hidden_flag,
                    follow_symlinks=False,
                )
                changed += 1
        except (AttributeError, OSError):
            continue
    return changed


def prepare_qt_runtime() -> None:
    """Подготовить bundled-плагины PySide6 на macOS.

    В некоторых каталогах macOS рекурсивно назначает содержимому `.venv`
    флаг UF_HIDDEN. QFactoryLoader не перечисляет такие dylib и аварийно
    завершает QApplication с сообщением об отсутствующем плагине `cocoa`.
    """
    if sys.platform != 'darwin':
        return

    try:
        import PySide6
    except ImportError:
        return

    plugin_root = Path(PySide6.__file__).resolve().parent / 'Qt' / 'plugins'
    platform_plugins = plugin_root / 'platforms'
    clear_hidden_flags(plugin_root)
    os.environ['QT_PLUGIN_PATH'] = str(plugin_root)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(platform_plugins)
