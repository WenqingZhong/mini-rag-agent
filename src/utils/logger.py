"""
Structured JSON logger.

Every log call produces a single-line JSON object so logs can be ingested
by tools like Datadog, Splunk, or CloudWatch without extra parsing.

Usage:
    logger = get_logger("my_module")
    logger.info("Something happened", extra={"key": "value"})
"""

import logging
import json
import sys
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """
    Custom log formatter that serialises each log record as JSON.

    Fields always present: timestamp, level, module, message.
    Any dict passed via extra={...} is merged in at the top level,
    making it easy to attach structured metadata to any log line.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        # Merge extra={...} fields — skip all standard LogRecord instance attributes
        # and private keys. We build the exclusion set from a real instance so that
        # instance-level attributes like 'args' and 'message' are also excluded
        # (Python 3.14 raises KeyError if you try to overwrite 'args').
        _reserved = set(
            logging.LogRecord(
                name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
            ).__dict__.keys()
        )
        for key, val in record.__dict__.items():
            if key not in _reserved and not key.startswith("_"):
                log_entry[key] = val

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with JSON formatting attached.

    Idempotent — calling this twice with the same name returns the same logger
    without adding duplicate handlers (guarded by the `if not logger.handlers` check).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False  # prevent double-logging via the root logger
    logger.setLevel(logging.DEBUG)
    return logger
