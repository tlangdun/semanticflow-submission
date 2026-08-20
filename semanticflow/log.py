"""Structured logging for SemanticFlow.

Provides consistent logging across all modules with configurable format and level.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter with colors for terminals."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",   # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True) -> None:
        super().__init__()
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = record.getMessage()
        
        # Include extra data if present
        extra = ""
        if hasattr(record, "extra_data") and record.extra_data:
            extra = f" | {record.extra_data}"
        
        return f"{timestamp} {level_str} [{record.name}] {message}{extra}"


class SemanticFlowLogger:
    """Logger wrapper with convenience methods for structured logging."""

    def __init__(self, name: str, level: str = "INFO", format: str = "text") -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        # Remove existing handlers
        self._logger.handlers.clear()
        
        # Add handler with appropriate formatter
        handler = logging.StreamHandler(sys.stderr)
        if format.lower() == "json":
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(TextFormatter())
        self._logger.addHandler(handler)
        
        # Prevent propagation to root logger
        self._logger.propagate = False

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, kwargs)

    def _log(self, level: int, message: str, extra: dict[str, Any]) -> None:
        record = self._logger.makeRecord(
            self._logger.name,
            level,
            "",
            0,
            message,
            (),
            None,
        )
        if extra:
            record.extra_data = extra
        self._logger.handle(record)


# Global logger instances cache
_loggers: dict[str, SemanticFlowLogger] = {}


def get_logger(
    name: str,
    level: str = "INFO",
    format: str = "text",
) -> SemanticFlowLogger:
    """Get or create a logger instance.
    
    Args:
        name: Logger name (typically module name)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Output format (text or json)
    
    Returns:
        SemanticFlowLogger instance
    """
    key = f"{name}:{level}:{format}"
    if key not in _loggers:
        _loggers[key] = SemanticFlowLogger(name, level, format)
    return _loggers[key]


def configure_logging(level: str = "INFO", format: str = "text") -> None:
    """Configure default logging settings for all SemanticFlow loggers.
    
    Args:
        level: Default log level
        format: Default output format
    """
    # Update existing loggers
    for key, logger in list(_loggers.items()):
        _, old_level, old_format = key.split(":")
        if old_level != level or old_format != format:
            new_key = f"{logger._logger.name}:{level}:{format}"
            _loggers[new_key] = SemanticFlowLogger(logger._logger.name, level, format)
            del _loggers[key]
