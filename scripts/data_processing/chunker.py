from langchain_text_splitters import NLTKTextSplitter

class TextSplitter:
    """
    A class to handle text splitting functionality using the NLTKTextSplitter.
    It splits documents into chunks based on specified chunk size and overlap.
    """

    def __init__(self, chunk_size: int=500, chunk_overlap: int=250):
        """
        Initializes the TextSplitter with specified chunk size and overlap.

        Parameters:
            chunk_size (int): The maximum size of each text chunk.
            chunk_overlap (int): The number of overlapping characters between consecutive chunks.
        """
        super().__init__()
        self.text_splitter = NLTKTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text):
        """
        Splits the provided document into chunks.

        Parameters:
            document (str): The document to be split into chunks.

        Returns:
            list: A list of text chunks.
        """
        chunks = self.text_splitter.split_text(text)
        return chunks
