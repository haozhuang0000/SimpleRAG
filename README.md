# SimpleRAG

SimpleRAG is a repository designed to demonstrate the use of Retrieval-Augmented Generation (RAG) with [Milvus](https://milvus.io/) and [LangChain](https://langchain.com/).

## Prerequisites

Before setting up SimpleRAG, ensure you have the following:

1. **LLamaParse API Key:**
   - Sign up and obtain your API key from [LLamaParse](https://cloud.llamaindex.ai/).

2. **Milvus Installation:**
   - Follow the official [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md) to set up a standalone Milvus instance using Docker.

3. **Environment Requirements:**
   - A machine with Python 3.11 installed.
   - *(Recommended)* A GPU server for hosting the embedding model & LLM.

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/haozhuang0000/SimpleRAG.git
    cd SimpleRAG
    ```

2. **Create a Conda Environment:**

    ```bash
    conda create -n simplerag python=3.11
    conda activate simplerag
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

### `.env` File Setup

Create a `.env` file in the root directory of the project and configure the following variables:

```env
VDB_HOST=YOUR_MILVUS_IP_ADDRESS
VDB_PORT=YOUR_MILVUS_PORT
EMBEDDING_HOST=YOUR_EMBEDDING_MODEL_IP_ADDRESS
EMBEDDING_PORT=YOUR_EMBEDDING_MODEL_PORT
OLLAMA_HOST=YOUR_OLLAMA_IP_ADDRESS
OLLAMA_PORT=YOUR_OLLAMA_PORT
LLAMAPARSER_API_KEY=your_llamaparse_api_key
```
## Usage

1. Create a folder `_static` & Put your PDF files under `_static`
2. `python main.py`

## RAG Workflow

The Retrieval-Augmented Generation (RAG) process in SimpleRAG follows these steps:

### 1. Create a Vector Database Collection

- **Initialize Collection:** Create a new collection in Milvus to store document embeddings.

### 2. Parse PDF Documents

- **Process PDFs:** Use LLamaParse to process all PDF files located in the `_static` directory.

### 3. Insert Data into Vector Database

#### 3.1 Split Documents

- **Document Chunks:** Break down parsed documents into chunks for processing.

#### 3.2 Embed Chunks

- **Generate Embeddings:** Use the embedding model to generate embeddings for each document chunk.

#### 3.3 Insert Embeddings

- **Store in Milvus:** Insert the generated embeddings into the Milvus vector database for future retrieval.

### 4. Query the Vector Database

#### 4.1 Embed Query

- **Query Embedding:** Convert the user's query into an embedding using the same embedding model.

#### 4.2 Search Database

- **Similarity Search:** Perform a similarity search in Milvus to find chunks that are most relevant to the query embedding.

#### 4.3 Retrieve Chunks

- **Fetch Results:** Retrieve the most relevant document chunks based on the similarity search results.

### 5. Generate Response with LLM

- **Contextual Response:** Utilize LangChain to generate a response from the language model, incorporating the retrieved context for an informed answer.
