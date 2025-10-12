import time
from loguru import logger
from typing import Any


def setup_logging(level: str = "INFO"):
    """Setup loguru logging configuration"""
    # Remove default handler
    logger.remove()
    
    # Add console handler with timestamp
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level.upper()
    )


def log_partial(partial_text: str):
    """Log ASR partial text"""
    if partial_text.strip():
        logger.info(f"ASR Partial: '{partial_text}'")

def log_asr_state(partial_text: str, rolling_text: str):
    """Log ASR state for debugging"""
    if partial_text.strip():
        logger.debug(f"ASR State - Partial: '{partial_text}', Rolling: '{rolling_text[:50]}...'")


def log_horizon(horizon_text: str):
    """Log LLM horizon prediction"""
    if horizon_text.strip():
        logger.info(f"LLM Horizon: '{horizon_text}'")


def log_hit_detection(hit_asr: bool, hit_llm: bool, partial_tail: str, horizon_text: str):
    """Log hit detection results"""
    logger.info(f"Hit Detection - ASR: {hit_asr}, LLM: {hit_llm}")
    if hit_asr or hit_llm:
        logger.warning(f"RISK DETECTED - Partial tail: '{partial_tail}', Horizon: '{horizon_text}'")


def log_daf_transition(active: bool, reason: str = ""):
    """Log DAF state transitions"""
    if active:
        logger.warning(f"DAF ACTIVATED - {reason}")
    else:
        logger.info(f"DAF DEACTIVATED - {reason}")


def log_latency(latency_ms: float):
    """Log estimated latency to DAF activation"""
    logger.info(f"DAF Latency: {latency_ms:.1f}ms")


def log_audio_status(status: str):
    """Log audio system status"""
    logger.info(f"Audio: {status}")


def log_error(error: str, details: Any = None):
    """Log error with optional details"""
    if details:
        logger.error(f"{error}: {details}")
    else:
        logger.error(error)


def get_timestamp_ms() -> float:
    """Get current timestamp in milliseconds"""
    return time.time() * 1000


def format_duration_ms(duration_ms: float) -> str:
    """Format duration in milliseconds as human readable string"""
    if duration_ms < 1000:
        return f"{duration_ms:.1f}ms"
    elif duration_ms < 60000:
        return f"{duration_ms/1000:.1f}s"
    else:
        return f"{duration_ms/60000:.1f}m"


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to maximum length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."
