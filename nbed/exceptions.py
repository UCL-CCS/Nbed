"""Custom Exceptions."""


class NbedConfigError(Exception):
    """Raised when config is not valid."""

    pass


class NbedDriverError(Exception):
    """Raise when NbedDriver finds itself in a bad state."""

    pass


class NbedLocalizerError(Exception):
    """Raise when Localizer sense check fails."""

    pass


class HamiltonianConverterError(Exception):
    """Base Exception class."""

    pass


class HamiltonianBuilderError(Exception):
    """Base Exception class."""

    pass
