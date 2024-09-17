import os
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, utility, Collection, MilvusClient
from langchain_core.runnables import ConfigurableField
from langchain_milvus import Milvus
from scripts.model.embedModel import NVEmbed
from scripts.logger.logger import Log
from scripts.logger.exceptions import MissingDBInfoError, CollectionNotFoundError

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class MilvusDB:
    # Class-level attributes
    conn = None

    # Retrieve host and port from environment variables
    host: str = os.getenv('VDB_HOST')
    port: str = os.getenv('VDB_PORT')

    # Initialize the embedding model
    embed_model: NVEmbed = NVEmbed()

    # Define keys for various fields in the collection
    ID_KEY: str = "pdf_id"
    VECTOR: str = "text_embedding"
    TEXT: str = "chunk_text"

    def __init__(self):
        """
        Initialize the MilvusDB instance by establishing a connection
        to the Milvus server and setting up the database.
        """
        super().__init__()
        self.logger = Log(f'{os.path.basename(__file__)}').getlog()
        if not MilvusDB.conn:
            if not self.host or not self.port:
                raise MissingDBInfoError()
            else:
                MilvusDB.conn = connections.connect(host=MilvusDB.host, port=MilvusDB.port)
                self.logger.info(f"Connected to Vector Database")
        # existing_db = db.list_database()
        # if MilvusDB.db_name not in existing_db:
        #     db.create_database(MilvusDB.db_name)
        #     self.logger.info(f"Successfully created database: {MilvusDB.db_name}")

        # db.using_database(MilvusDB.db_name)
        # if not connections.has_connection(self.db_name):
        #     connections.connect(alias=self.db_name, host=self.host, port=self.port)

    def load_collection(self, colname):
        Collection(colname).load()

    def is_collection_exists(self, col_name: str):
        """
        Check if a collection with the given name exists in the database.

        Args:
            col_name (str): The name of the collection to check.

        Returns:
            bool: True if the collection exists, False otherwise.
        """

        collections = utility.list_collections()
        for col in collections:
            if col == col_name:
                return True
        return False

    def create_collection(self, colname: str):
        """
        Create a new collection in the database with the specified schema.

        Args:
            colname (str): The name of the collection to create.
        """
        if not self.is_collection_exists(colname):

            pdf_id = FieldSchema(
                name=MilvusDB.ID_KEY,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            )

            pdf_name = FieldSchema(
                name="pdf_name",
                dtype=DataType.VARCHAR,
                max_length=200,
            )
            chunk_number = FieldSchema(
                name="chunk_number",
                dtype=DataType.INT64,
            )
            chunk_text = FieldSchema(
                name=MilvusDB.TEXT,
                dtype=DataType.VARCHAR,
                max_length=65000
            )
            text_embedding = FieldSchema(
                name=MilvusDB.VECTOR,
                dtype=DataType.FLOAT_VECTOR,
                dim=MilvusDB.embed_model.DIM
            )

            schema = CollectionSchema(
                fields=[pdf_id, pdf_name, chunk_number, chunk_text, text_embedding],
                description="Embed pdf file"
            )

            collection = Collection(colname, schema)

            # Create an index for the text_embedding field
            index_params = {
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
                "metric_type": "L2"
            }
            collection.create_index(field_name="text_embedding", index_params=index_params)

    def insert_collection(self, colname: str, data: list):
        """
        Insert data into the specified collection.

        Args:
            colname (str): The name of the collection to insert data into.
            data (dict): A dictionary containing the data to insert.
                         Expected keys: 'text', 'pdf_name', 'chunk_number'
        """

        # self.logger.info(f"{filename} - {chunk} is being inserted to: " + colname)
        col = Collection(colname)
        col.load()
        col.insert(data)

    def check_existing_file(self, colname: str, pdf_name: str):

        collection = Collection(colname)
        self.load_collection(colname)
        query_expr = f'pdf_name == "{pdf_name}"'
        search_results = collection.query(expr=query_expr)
        if len(search_results) == 0:
            return False
        else:
            self.logger.info(f'{pdf_name} is already exist in our vector database')
            return True

    def drop_collection(self, colname):

        utility.drop_collection(colname)

class MilvusWLangChain(Milvus):

    """
    A class that integrates Milvus with Langchain to provide a configurable retriever.

    Attributes:
        milvus (Milvus): An instance of the Milvus vector store.
    """

    def __init__(self, colname: str, pdf_name: str, *args, **kwargs):
        if not colname:
            raise CollectionNotFoundError(colname)
        # Initialize the Milvus vector store with Langchain compatibility
        self.milvus = Milvus(
            embedding_function=MilvusDB.embed_model,
            collection_name=colname,
            connection_args={
                "host": MilvusDB.host,
                "port": MilvusDB.port
            },
            vector_field=MilvusDB.VECTOR,
            primary_field=MilvusDB.ID_KEY,
            text_field="chunk_text",
            *args,
            **kwargs
        )
        self.pdf_name = pdf_name

    def get_retriever(self, relavant_chunk_size):
        """
        Configures and returns a retriever with specified filters.

        Returns:
            Retriever: A configured retriever instance.
        """
        return self.milvus.as_retriever().configurable_fields(
            search_kwargs=ConfigurableField(
                id="expr_filter",
            )
        ).with_config(
            configurable={
                "expr_filter": {
                    "expr": f"pdf_name == '{self.pdf_name}'",
                    "k": relavant_chunk_size,
                }
            }
        )

if __name__ == '__main__':

    milvusdb = MilvusDB()
    milvusdb.drop_collection('col_test')


