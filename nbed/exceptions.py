"""Custom Exceptions."""


class NbedDriverError(Exception):
    """Raise when NbedDriver finds itself in a bad state."""

    pass


class NbedLocalizerError(Exception):
    """Raise when Localizer sense check fails."""

    pass


class HamiltonianBuilderError(Exception):
    """Base Exception class."""

    pass
