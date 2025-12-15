"""
Intelligent Document Assistant - AI-powered document Q&A system

A modern Python framework combining LangChain, LLMs, and vector databases
to create intelligent document analysis and retrieval systems.
"""

__version__ = "0.1.0"
__author__ = "Nikhil Budhiraja"
__email__ = "nikhilbudhiraja002@gmail.com"

from config import get_config, Config

__all__ = [
    "get_config",
    "Config",
]
