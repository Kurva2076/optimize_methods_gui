import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from lib.data.xyz_loader import XYZLoader
from lib.data.xyz_saver import save_points_to_xyz
from lib.optimization.filtering import clean, ALS_PRESET


class OptimizationWorker(QObject):
    progress = Signal(str, int, int)
    file_processed = Signal(str, float)
    finished = Signal(int, int, str)
    failed = Signal(str)

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        exclude_files: list[Path] | None = None,
    ) -> None:
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.exclude_files = exclude_files or []

    @Slot()
    def run(self) -> None:
        try:
            files = sorted(
                path for path in self.input_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() == '.xyz'
                and path not in self.exclude_files
            )

            if not files:
                raise ValueError(
                    'В папке нет доступных файлов .xyz для обработки.'
                )

            self.output_dir.mkdir(parents=True, exist_ok=True)

            success_count = 0
            error_messages = []

            for index, file_path in enumerate(files, start=1):
                started_at = time.perf_counter()

                self.progress.emit(
                    f'Обработка: {file_path.name}',
                    index - 1,
                    len(files)
                )

                try:
                    classes, points = XYZLoader(
                        str(file_path)
                    ).load(dtype=np.float64)

                    result = clean(
                        points,
                        classes,
                        steps=ALS_PRESET
                    )

                    clean_classes = classes[result.keep]
                    clean_points = points[result.keep]

                    clean_path = (
                        self.output_dir /
                        f'{file_path.stem}_clean.xyz'
                    )

                    save_points_to_xyz(
                        points=clean_points,
                        filepath=str(clean_path),
                        classes=clean_classes,
                    )

                    noise_mask = ~result.keep

                    if np.any(noise_mask):
                        noise_path = (
                            self.output_dir /
                            f'{file_path.stem}_noise.xyz'
                        )

                        save_points_to_xyz(
                            points=points[noise_mask],
                            filepath=str(noise_path),
                            classes=result.codes[noise_mask],
                        )

                    success_count += 1

                except Exception as error:
                    error_messages.append(
                        f'{file_path.name}: {error}'
                    )

                finally:
                    elapsed = time.perf_counter() - started_at
                    self.file_processed.emit(
                        str(file_path),
                        elapsed
                    )

                self.progress.emit(
                    f'Обработано файлов: {index} из {len(files)}',
                    index,
                    len(files)
                )

            report = (
                '\n\nОшибки:\n' + '\n'.join(error_messages)
                if error_messages else ''
            )

            self.finished.emit(
                success_count,
                len(files),
                report
            )

        except Exception as error:
            self.failed.emit(str(error))
