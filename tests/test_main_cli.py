from pathlib import Path

from PySide6.QtWidgets import QApplication

from main import apply_initial_directories, parse_application_arguments
from main_window import MainWindow
from screens.preprocessing import PreprocessingScreen


def test_parse_application_arguments_accepts_initial_directories(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / 'input'
    output_directory = tmp_path / 'output'
    input_directory.mkdir()
    output_directory.mkdir()

    arguments, qt_arguments = parse_application_arguments(
        [
            '--input-dir',
            str(input_directory),
            '--output-dir',
            str(output_directory),
            '-style',
            'fusion',
        ]
    )

    assert arguments.input_dir == input_directory.resolve()
    assert arguments.output_dir == output_directory.resolve()
    assert qt_arguments == ['-style', 'fusion']


def test_apply_initial_directories_fills_preprocessing_screen(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    input_directory = tmp_path / 'input'
    output_directory = tmp_path / 'output'
    input_directory.mkdir()
    output_directory.mkdir()
    (input_directory / 'cloud.xyz').write_text('0 0 0 0\n', encoding='utf-8')

    window = MainWindow()
    apply_initial_directories(window, input_directory, output_directory)
    preprocessing = next(
        screen
        for screen in window.screens
        if isinstance(screen, PreprocessingScreen)
    )

    assert preprocessing.input_dir == str(input_directory)
    assert preprocessing.output_dir == str(output_directory)
    assert [path.name for path in preprocessing.all_files] == ['cloud.xyz']
    assert preprocessing.files_count_label.text() == 'Выбранные файлы (1)'
    assert preprocessing.output_label.text() == str(output_directory)

    window.deleteLater()
    app.processEvents()
