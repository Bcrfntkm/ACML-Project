"""Configuration loader for the chemical annotation agent.

Reads ``Science/.env`` (via python-dotenv) and ``Science/config.yaml``
(via PyYAML) and exposes a :class:`Settings` dataclass with all runtime
parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "python-dotenv is required. Install with: pip install python-dotenv"
    ) from exc

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required. Install with: pip install pyyaml"
    ) from exc

# ---------------------------------------------------------------------------
# Resolve paths relative to this file's location
# ---------------------------------------------------------------------------
_AGENT_DIR = Path(__file__).parent          # Science/agent/
_SCIENCE_DIR = _AGENT_DIR.parent            # Science/
_ENV_FILE = _SCIENCE_DIR / ".env"
_CONFIG_FILE = _SCIENCE_DIR / "config.yaml"
_PARSER_DIR = _SCIENCE_DIR / "parser" / "ReactionMiner"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict (empty dict on failure)."""
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except Exception:  # pragma: no cover
        return {}


@dataclass
class Settings:
    """All runtime settings for the annotation pipeline."""

    # LLM / API
    api_key: str = ""
    base_url: str = "https://openai.api.proxyapi.ru/v1"
    model: str = "deepseek/deepseek-v3"
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout: int = 60

    # RAG
    rag_enabled: bool = False
    rag_top_k: int = 5
    rag_index_dir: str = "Science/rag_index"

    # Parser
    parser_dir: str = field(default_factory=lambda: str(_PARSER_DIR.resolve()))

    # Prompts
    prompt_version: str = "v2"


def load_settings() -> Settings:
    """Load settings from ``.env`` + ``config.yaml``, return a :class:`Settings` instance.

    Priority (highest → lowest):
    1. Environment variables (already set before this call, or loaded from .env)
    2. ``Science/config.yaml``
    3. Dataclass defaults
    """
    # 1. Load .env into the process environment (does not override existing vars)
    if _ENV_FILE.is_file():
        load_dotenv(dotenv_path=_ENV_FILE, override=False)

    # 2. Parse config.yaml
    cfg = _load_yaml(_CONFIG_FILE)
    api_cfg: dict = cfg.get("api", {})
    rag_cfg: dict = cfg.get("rag", {})

    # 3. Build Settings, preferring env vars over yaml values over defaults
    api_key: str = os.environ.get("PROXYAPI_KEY", "") or str(api_cfg.get("api_key", ""))
    base_url: str = "https://openai.api.proxyapi.ru/v1"  # always use ProxyAPI
    model: str = str(api_cfg.get("model", "deepseek/deepseek-v3"))
    max_tokens: int = int(api_cfg.get("max_tokens", 2048))
    temperature: float = float(api_cfg.get("temperature", 0.0))
    timeout: int = int(api_cfg.get("timeout", 60))

    rag_enabled: bool = bool(rag_cfg.get("enabled", False))
    rag_top_k: int = int(rag_cfg.get("top_k", 5))
    rag_index_dir: str = str(rag_cfg.get("index_dir", "Science/rag_index"))

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        rag_enabled=rag_enabled,
        rag_top_k=rag_top_k,
        rag_index_dir=rag_index_dir,
        parser_dir=str(_PARSER_DIR.resolve()),
    )


# Module-level singleton — imported by other modules as ``from .config import settings``
settings: Settings = load_settings()
