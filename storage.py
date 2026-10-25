"""Сохранение параметров подбора. Пока — простой JSON-файл (формат будет уточнён)."""
import json
from datetime import datetime
from pathlib import Path

from hyperparams import SearchConfig
from resources import DATA_DIR

SEARCH_CONFIG_PATH = DATA_DIR / 'hyperparameter_search.json'


def save_search_config(config: SearchConfig, path: Path = SEARCH_CONFIG_PATH) -> Path:
    """Записывает конфигурацию подбора в JSON и возвращает путь к файлу."""
    payload = {'created_at': datetime.now().isoformat(timespec='seconds'), **config.to_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path
