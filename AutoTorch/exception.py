"""
Contains exceptions for the AutoTorch package.
"""
class ConfigError(Exception):
    """
    Any error related to the configuration
    """

    pass


class ConfigWarning(Warning):
    """
    Any warning related to the configuration
    """

    pass


class CheckPointException(Exception):
    """
    Related to errors with the Trainer.
    """

    pass