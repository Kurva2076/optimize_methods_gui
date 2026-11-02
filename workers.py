import json
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from lib.pipeline import iter_process_directory


class OptimizationWorker(QObject):
    progress = Signal(str, int, int)
    file_processed = Signal(str, float)
    finished = Signal(int, int, str, str)
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
            excluded = {
                path.resolve()
                for path in self.exclude_files
            }

            files_count = sum(
                1
                for path in self.input_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".xyz"
                and path.resolve() not in excluded
            )

            results = []
            error_messages = []
            success_count = 0
            started_at = time.perf_counter()

            iterator = iter_process_directory(
                input_dir=self.input_dir,
                export_dir=self.output_dir,
                debug=False,
            )

            for index, result in enumerate(iterator, start=1):
                elapsed = sum(result.timings.values())

                self.file_processed.emit(
                    str(result.input_path),
                    elapsed,
                )

                if result.success:
                    success_count += 1
                else:
                    error_messages.append(
                        f"{result.input_path.name}: {result.error}"
                    )

                file_result = result.to_dict()
                file_result["output_path"] = (
                    str(result.output_path)
                    if result.output_path else None
                )
                file_result["timings"] = result.timings

                results.append(file_result)

                self.progress.emit(
                    f"Обработано файлов: {index} из {files_count}",
                    index,
                    files_count,
                )

            elapsed_seconds = time.perf_counter() - started_at

            report = (
                "\n\nОшибки:\n" + "\n".join(error_messages)
                if error_messages else ""
            )

            json_report = json.dumps(
                {
                    "success": success_count == files_count,
                    "input_dir": str(self.input_dir.resolve()),
                    "export_dir": str(self.output_dir.resolve()),
                    "elapsed_seconds": elapsed_seconds,
                    "files": results,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

            self.finished.emit(
                success_count,
                files_count,
                report,
                json_report,
            )

        except Exception as error:
            self.failed.emit(str(error))
