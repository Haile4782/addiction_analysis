from loguru import logger

logger.add("api.log", rotation="1 MB")