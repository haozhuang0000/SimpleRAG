from rag.VDB_Common import *
from rag.Prompts import FinancialExpertPrompt
from data_processing.parser import PDFParser
from data_processing.chunker import TextSplitter
from model.embedModel import NVEmbed
from logger.exceptions import MissingDBInfoError, CollectionNotFoundError, ModelLoadingWarning

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag.llm import Llm
import warnings

class SimpleRAG(
    MilvusDB,
    PDFParser,
    NVEmbed,
    FinancialExpertPrompt
):

    def __init__(self, model) -> None:
        super().__init__()
        self.logger = Log(f'{os.path.basename(__file__)}').getlog()
        self.model = model
        ModelLoadingWarning(self.model)
        self.llm = Llm(model=self.model)
        self.QA_CHAIN_PROMPT = self.get_prompt_template()

    def insert_VDB(self, colname: str, document_path: str):

        if not self.is_collection_exists(colname):
            raise CollectionNotFoundError(colname)

        documents = self.load_directory(document_path, limit=3)
        text_splitter = TextSplitter(chunk_size=500, chunk_overlap=250)

        for document in documents:
            pdf_name = document['pdf_name'].replace('.pdf', '')
            pdf_text = document['pdf_text']
            if self.check_existing_file(colname=colname, pdf_name=pdf_name):
                continue

            self.logger.info(f'Start processing file: {pdf_name}')
            chunks = text_splitter.split_text(pdf_text)
            for i, chunk in enumerate(chunks):
                chunk_vector = self.embed_documents([chunk])
                data = [
                    [pdf_name], # pdf_name
                    [i], # chunk_number
                    [chunk], # chunk_text
                    chunk_vector # chunk_vector
                ]
                self.insert_collection(colname, data)
            self.logger.info(f'Done processing file - {pdf_name}')

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def query_VDB(self, colname: str, user_query: str, pdf_name: str):

        if not self.is_collection_exists(colname):
            raise CollectionNotFoundError(colname)

        milvusWlangchain = MilvusWLangChain(colname=colname, pdf_name=pdf_name)

        self.logger.info('Retrieving relavant chunks from vector database...')
        retriever = milvusWlangchain.get_retriever(relavant_chunk_size=6)
        results = retriever.invoke(
            user_query
        )
        if results == []:
            warnings.warn("It does not retrieve any information from the vector database, "
                          "please check your collection in vdb!")
        formatted_context = self.format_docs(results)
        self.logger.info('Relevant chunks have been retrieved')


        rag_chain = (
                {"context": RunnablePassthrough(), "input": RunnablePassthrough()}
                | self.QA_CHAIN_PROMPT
                | self.llm
                | StrOutputParser()
        )
        rag_chain_input = {"context": formatted_context, "input": user_query}
        reply = rag_chain.invoke(rag_chain_input)
        return reply

    def run(self,
            colname: str,
            document_path: str,
            user_query: str,
            pdf_name: str):

        ############################### -- Create VDB -- ###############################

        if not self.is_collection_exists(colname):
            self.logger.info(f'{colname} is not exist in our database, creating...')
            self.create_collection(colname)
        else:
            self.logger.info(f'{colname} existed in our database, proceed to next operation')

        ############################### -- Insert to VDB -- ###############################
        self.logger.info('Start running insertion function...')
        self.insert_VDB(colname, document_path)
        self.logger.info('Done insertion')

        ############################### -- search VDB and reply -- ###############################
        self.logger.info('Start querying our VDB, and getting reply from LLM')
        reply = self.query_VDB(colname, user_query, pdf_name)
        self.logger.info(
            f'Reply from {self.model} successfully generated: \n'
            f'{reply}'
        )



if __name__ == '__main__':

    simplerag = SimpleRAG('llama3.1:70b-instruct-q4_0')
    simplerag.run(
        colname='col_test',
        document_path= '../_static',
        user_query='what is actuarial spread?',
        pdf_name='Actuarial Spread White Paper'
    )