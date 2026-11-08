import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from main_window import MainWindow
from qt_runtime import prepare_qt_runtime
from resources import STYLE_FILES
from screens.preprocessing import PreprocessingScreen


def load_styles() -> str:
    return '\n'.join(
        path.read_text(encoding='utf-8') for path in STYLE_FILES if path.exists()
    )


def existing_directory(value: str) -> Path:
    directory = Path(value).expanduser().resolve()
    if not directory.is_dir():
        raise argparse.ArgumentTypeError(f'Папка не найдена: {directory}')
    return directory


def parse_application_arguments(
    arguments: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description='PointCloud — классификация облаков точек',
    )
    parser.add_argument(
        '--input-dir',
        type=existing_directory,
        help='заранее выбрать папку с исходными файлами .xyz',
    )
    parser.add_argument(
        '--output-dir',
        type=existing_directory,
        help='заранее выбрать существующую папку для результатов',
    )
    return parser.parse_known_args(arguments)


def apply_initial_directories(
    window: MainWindow,
    input_directory: Path | None,
    output_directory: Path | None,
) -> None:
    preprocessing = next(
        (
            screen
            for screen in window.screens
            if isinstance(screen, PreprocessingScreen)
        ),
        None,
    )
    if preprocessing is None:
        return
    if input_directory is not None:
        preprocessing.load_directory(str(input_directory))
    if output_directory is not None:
        preprocessing.load_output_directory(str(output_directory))


def main() -> None:
    prepare_qt_runtime()
    arguments, qt_arguments = parse_application_arguments(sys.argv[1:])
    app = QApplication([sys.argv[0], *qt_arguments])
    app.setStyleSheet(load_styles())
    window = MainWindow()
    apply_initial_directories(
        window,
        arguments.input_dir,
        arguments.output_dir,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
