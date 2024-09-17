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
    def __init__(self, collection_name):
        message = f"Collection '{collection_name}' does not exist. Please create the collection first."
        super().__init__(message)
