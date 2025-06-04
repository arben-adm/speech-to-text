import logging
from typing import Optional, Union
import os

class AppLogger:
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File Handler
        if log_file:
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler(f"logs/{log_file}")
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)
            
        self.logger.addHandler(console_handler)
        
    def info(self, message: str) -> None:
        self.logger.info(message)
        
    def error(self, message: str) -> None:
        self.logger.error(message)
        
    def warning(self, message: str) -> None:
        self.logger.warning(message)
        
    def debug(self, message: str) -> None:
        self.logger.debug(message)


# Cache for loggers to avoid creating multiple instances
_loggers = {}

def get_logger(name: str, log_file: Optional[str] = None) -> AppLogger:
    """
    Get or create a logger instance
    
    Args:
        name: Name of the logger (usually __name__)
        log_file: Optional log file name
        
    Returns:
        AppLogger instance
    """
    global _loggers
    if name not in _loggers:
        _loggers[name] = AppLogger(name, log_file)
    return _loggers[name]