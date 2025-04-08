import logging
from typing import Optional
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