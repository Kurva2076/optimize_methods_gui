## Репозиторий серой команды (GUI приложение для методов оптимизации)

Подробнее про методы оптимизации: https://github.com/d2min/optimyze_methods/

ВРЕМЕННАЯ МЕРА:
- на данный момент репозиторий методов оптимизации не предназначен для установки скриптов в качестве библиотеки
- для работы GUI репозитория необходимо выполнить следующие команды:
```commandline
cd ..
```
```commandline
git clone https://github.com/d2min/optimyze_methods/
```
```commandline
cd optimyze_methods
```
```commandline
python -m pip install -e .
```

Идеальная реализация:
- необходимо попросить красную команду добавить `pyproject.toml` в корень проекта с таким содержимым:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "optimyze-methods"
version = "0.1.0"
description = "Методы оптимизации данных облака точек"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "open3d>=0.17.0",
    "scipy>=1.10",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["lib*"]
namespaces = true
```
- после добавления этого файла необходимо заменить в данном репозитории в файле `requirements.txt` "-e ../optimyze_methods" на "git+https://github.com/d2min/optimyze_methods.git"

Установка зависимостей:
```commandline
python -m pip install -r requirements.txt
```
