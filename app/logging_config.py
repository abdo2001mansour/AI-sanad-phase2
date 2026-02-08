"""
Custom logging configuration to suppress socket errors from client disconnections
and capture logs in memory for the logs endpoint.
"""
import logging
from collections import deque
from datetime import datetime
from typing import List, Dict, Any
import threading


# Thread-safe in-memory log buffer
class LogBuffer:
    """Thread-safe circular buffer for storing recent log entries"""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, record: Dict[str, Any]):
        with self.lock:
            self.buffer.append(record)
    
    def get_logs(self, count: int = 100) -> List[Dict[str, Any]]:
        with self.lock:
            logs = list(self.buffer)
            return logs[-count:] if count < len(logs) else logs
    
    def clear(self):
        with self.lock:
            self.buffer.clear()


# Global log buffer instance
log_buffer = LogBuffer(max_size=1000)


class MemoryLogHandler(logging.Handler):
    """Custom handler that stores logs in memory"""
    
    def __init__(self, buffer: LogBuffer):
        super().__init__()
        self.buffer = buffer
    
    def emit(self, record):
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                import traceback
                log_entry["exception"] = ''.join(traceback.format_exception(*record.exc_info))
            
            self.buffer.add(log_entry)
        except Exception:
            pass  # Don't crash on logging errors


class SupressSocketErrors(logging.Filter):
    """Filter to suppress socket.send() errors from client disconnections"""
    
    def filter(self, record):
        # Suppress the common socket errors that occur when client disconnects
        error_messages = [
            "socket.send() raised exception",
            "ConnectionResetError",
            "BrokenPipeError",
            "Connection reset by peer",
            "Broken pipe"
        ]
        
        message = record.getMessage()
        
        # Don't log if it's one of these errors
        for error_msg in error_messages:
            if error_msg in message:
                return False
        
        return True


def configure_logging():
    """Apply custom logging filters and memory handler"""
    # Create memory handler with formatter
    memory_handler = MemoryLogHandler(log_buffer)
    memory_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    memory_handler.setFormatter(formatter)
    
    # Get the root logger and add memory handler
    root_logger = logging.getLogger()
    root_logger.addHandler(memory_handler)
    root_logger.addFilter(SupressSocketErrors())
    
    # Set root logger level to capture all logs
    if root_logger.level == logging.NOTSET:
        root_logger.setLevel(logging.INFO)
    
    # Get the uvicorn error logger
    uvicorn_error = logging.getLogger("uvicorn.error")
    uvicorn_error.addFilter(SupressSocketErrors())
    uvicorn_error.addHandler(memory_handler)
    
    # Get the uvicorn access logger
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.addFilter(SupressSocketErrors())
    uvicorn_access.addHandler(memory_handler)
    
    # Also capture app loggers
    app_logger = logging.getLogger("app")
    app_logger.addHandler(memory_handler)
    app_logger.setLevel(logging.DEBUG)


def get_recent_logs(count: int = 100) -> List[Dict[str, Any]]:
    """Get the most recent log entries"""
    return log_buffer.get_logs(count)


def clear_logs():
    """Clear the log buffer"""
    log_buffer.clear()
