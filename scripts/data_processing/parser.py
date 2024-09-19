import os
import time

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from scripts.logger.logger import Log
from collections import defaultdict
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


class PDFParser:
    """
    PDFParser class to handle parsing of PDF files using LlamaParse.
    It can process individual PDF files or load all PDF files in a directory.
    """

    # API key for the LlamaParse service
    parser_api_key = os.getenv('LLAMAPARSER_API_KEY')

    def __init__(self):
        """
        Initializes the PDFParser with a logger and an instance of LlamaParse.
        """
        super().__init__()
        # Set up logger
        self.logger = Log(f'{os.path.basename(__file__)}').getlog()

        # Initialize the LlamaParse data_processing with the API key and options
        self.parser = LlamaParse(
            api_key=self.parser_api_key,
            result_type="markdown",
            verbose=True
        )

    def load_directory(self, directory_path: str, limit: int):
        """
        Loads and parses all PDF files in the specified directory.

        Args:
            directory_path (str): Path to the directory containing PDF files.

        Returns:
            Parsed data from all PDF files in the directory.
        """
        self.logger.info(f'Loading all PDF files in directory, path: {directory_path}')
        current_time = time.time()
        file_extractor = {".pdf": self.parser}
        documents = SimpleDirectoryReader(directory_path, file_extractor=file_extractor, num_files_limit=limit).load_data()
        grouped_documents = defaultdict(list)
        for doc in documents:
            filename = doc.metadata['file_name']  # Assuming the metadata contains the filename
            grouped_documents[filename].append(doc)

        all_docs = []
        for key, values in grouped_documents.items():
            result_dict = {'pdf_name': key, 'pdf_text': None}
            texts = []
            for value in values:
                texts.append(value.text)
            result_dict['pdf_text'] = '\n'.join(texts)
            all_docs.append(result_dict)
        end_time = time.time()
        seconds = end_time - current_time
        self.logger.info(f'It takes: {seconds}s to process files in path: {directory_path}')
        return all_docs

