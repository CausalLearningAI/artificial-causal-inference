"""Factory for creating domain handlers"""
from typing import Type
from pathlib import Path

from .base import DomainHandler, ExperimentConfig
from .ants import AntsHandler
from .mice import MiceHandler


DOMAIN_REGISTRY = {
    'ants': AntsHandler,
    'mice': MiceHandler,
}


def get_domain_handler(config: ExperimentConfig) -> DomainHandler:
    """
    Factory function to create appropriate domain handler
    
    Args:
        config: ExperimentConfig with domain and version
        
    Returns:
        DomainHandler subclass instance
        
    Raises:
        ValueError: If domain is not registered
    """
    handler_class = DOMAIN_REGISTRY.get(config.domain)
    if handler_class is None:
        raise ValueError(
            f"Unknown domain: {config.domain}. "
            f"Available domains: {', '.join(DOMAIN_REGISTRY.keys())}"
        )
    return handler_class(config)


def register_domain(domain_name: str, handler_class: Type[DomainHandler]):
    """
    Register a new domain handler
    
    Args:
        domain_name: Name of the domain
        handler_class: Handler class (must inherit from DomainHandler)
    """
    if not issubclass(handler_class, DomainHandler):
        raise TypeError(f"{handler_class} must inherit from DomainHandler")
    DOMAIN_REGISTRY[domain_name] = handler_class
