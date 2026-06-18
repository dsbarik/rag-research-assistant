class ServiceException(Exception):
    """Base exception for service layer errors."""
    pass

class ResourceNotFoundException(ServiceException):
    """Raised when a requested resource is not found."""
    pass

class InvalidRequestException(ServiceException):
    """Raised when a request is invalid."""
    pass
