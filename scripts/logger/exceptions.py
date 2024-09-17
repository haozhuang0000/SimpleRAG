# exceptions.py
class VDBException(Exception):
    """Base exception for VDB-related errors."""
    pass

class MissingDBInfoError(VDBException):
    """Exception raised when database information is missing."""
    def __init__(self, message="Environment variables VDB_HOST and VDB_PORT must be set."):
        super().__init__(message)

class CollectionNotFoundError(VDBException):
    """Exception raised when a specified collection does not exist."""
    def __init__(self, collection_name: str):
        message = f"Collection '{collection_name}' does not exist. Please create the collection first."
        super().__init__(message)

class ModelNotFoundError(VDBException):
    """Exception raised when a specified collection does not exist."""
    def __init__(self, model: str, available_models: list[str]):
        message = (
            f"Model '{model}' does not exist. "
            f"Available models: {', '.join(available_models)}. "
            "If you want to use a different model, please contact your server administrator."
        )
        super().__init__(message)

class ModelLoadingWarning(VDBException):
    """Exception raised when a specified collection does not exist."""
    def __init__(self, model: str):
        if model == 'llama3.1:70b-instruct-q4_0':
            message = (
                f"Loading model '{model}' may require a longer response time"
            )
            super().__init__(message)
        else:
            pass