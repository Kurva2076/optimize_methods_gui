## Репозиторий серой команды (GUI приложение для методов оптимизации)

Подробнее про методы оптимизации: https://github.com/d2min/optimyze_methods/

Установка зависимостей:
```commandline
python -m pip install -r requirements.txt
```

## Просмотр результатов обучения

Экран «Обучение» читает готовые артефакты PointNet++ из репозитория
`lidar-segmentation-kubsu/mihail-pointer-experiment`. Нажмите
«Выбрать результаты» и укажите папку конкретного запуска `runs/<run_id>`
или вложенную папку `results`.

Ожидаемая структура:

```text
runs/<run_id>/
├── history.csv
├── config.json
├── metadata.json
└── results/
    ├── metrics.json
    ├── metrics.csv
    └── per_scene_metrics.csv
```

Интерфейс строит четыре графика из `history.csv`, таблицы метрик из CSV/JSON
и матрицу ошибок из `metrics.json`. PyTorch и файл весов `best_pointnet2.pt`
для просмотра результатов не требуются. Все пути обрабатываются через
`pathlib`, поэтому можно выбирать обычные папки Windows.

### Быстрая ручная проверка интерфейса

В репозитории есть небольшой демонстрационный запуск
`example/training_run_demo`. Он нужен только для проверки интерфейса и не
выдаётся за результат реального обучения.

Обычный сценарий:

1. Запустите приложение командой `python main.py`.
2. Нажмите шаг `3 — Обучение` в верхней части окна.
3. Нажмите `Выбрать результаты`.
4. Выберите папку `example/training_run_demo`.
5. Переключите вкладки `Графики`, `Метрики`, `Матрица ошибок`. На вкладке
   `Графики` проверьте обе круглые стрелки: они циклически переключают четыре
   реальных представления данных из `history.csv`.

Чтобы сразу открыть заполненный экран без диалога выбора папки:

```commandline
python scripts/run_training_demo.py
```

На Windows PowerShell после клонирования проекта:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_training_demo.py
```

Для проверки загрузчика можно также выбрать папку `results` внутри
демонстрационного запуска. Если выбрать произвольную папку, интерфейс должен
показать понятное сообщение об отсутствующих `history.csv` или `metrics.json`.

Проверки разработчика:

```commandline
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check training_results.py screens/training.py tests
```
