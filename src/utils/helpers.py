import logging
import logging.handlers
import os
from pathlib import Path
from typing import Any, Dict, List


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(log_file: str = "./logs/app.log", level: str = "INFO") -> None:
    os.makedirs(Path(log_file).parent, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        ),
    ]
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt, handlers=handlers)


def ensure_directories(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def format_sources_markdown(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    seen = set()
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        page = meta.get("page", "N/A")
        year = meta.get("year", "")
        key = f"{source}_{page}"
        if key in seen:
            continue
        seen.add(key)
        year_str = f" ({year})" if year and year != "unknown" else ""
        lines.append(f"**[{i}]** {source}{year_str} — Page {page}")
    return "\n".join(lines)
