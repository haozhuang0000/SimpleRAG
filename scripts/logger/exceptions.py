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

class SimpleRagWarning():

    WarningModel = 'llama3.1:70b-instruct-q4_0'
    WarningModelMSG = f"Loading model '{WarningModel}' may require a longer response time."
    WarningQueryResult = "It does not retrieve any information from the vector database, please check your collection in vdb!"