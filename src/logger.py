# src/logger.py
import os
from datetime import datetime

LOG_FILE_PATH = "event_log.txt"

def log_event(message):
    """Registra eventos com timestamp no arquivo de log"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE_PATH, "a") as file:
        file.write(f"[{timestamp}] {message}\n")
    print(f"[LOG] {message}")  # Exibe no terminal também para monitoramento
