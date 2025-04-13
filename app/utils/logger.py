# app/utils/logger.py

import logging

import coloredlogs
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


field_styles = {
    "levelname": {"color": "cyan", "bold": True},
    "filename": {"color": "magenta"},
    "message": {"color": "white"},
}


log_format = "%(levelname)s | (%(filename)s): %(message)s"

coloredlogs.install(
    level=settings.LOG_LEVEL,
    logger=logger,
    fmt=log_format,
    field_styles=field_styles,
)
