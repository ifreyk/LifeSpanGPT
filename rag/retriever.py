from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.vectorstores import FAISS
from langchain.retrievers.document_compressors import CohereRerank
from pydantic import BaseModel, Field
from typing import List, Union
import torch

if torch.backends.mps.is_available(): # For macOS devices
    DEVICE = "mps"
elif torch.cuda.is_available(): # For NVIDIA GPUs
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

class RetrieverConfig(BaseModel):
    """
    A configuration model for the Retriever class, which stores parameters required for the retrieval process.

    Attributes:
        file_path (str): The path to the file to be processed.
        embeding_model (str): The name of the embedding model to be used.
        reranker_model (str): The name of the reranker model.
        chunk_size (int): The size of chunks for splitting the documents.
        chunk_overlap (int): The overlap size for splitting documents.
        COHERE_TOKEN (str): The API key for accessing the Cohere service.
    """
    file_path: Union[str, None] = Field(default=None)
    embeding_model: Union[str, None] = Field(default=None)
    reranker_model: Union[str, None] = Field(default=None)
    chunk_size: Union[int, None] = Field(default=None)
    chunk_overlap: Union[int, None] = Field(default=None)
    COHERE_TOKEN: Union[str, None] = Field(default=None)


class Retriever:
    """
    A class to perform document retrieval, embedding, and compression using different models.

    Attributes:
        config: The configuration object containing parameters for the retrieval process.

    Methods:
        __init__(config: RetrieverConfig):
            Initializes the Retriever instance with the provided configuration.
        
        split_documents():
            Loads and splits documents from the specified file path into chunks based on the chunk size and overlap.
        
        create_embeding_model():
            Creates and returns an embedding model using the configuration provided.
        
        create__basic_retriever(docs, embeddings):
            Creates and returns a basic retriever using the given documents and embeddings.
        
        create_compression_retriever(retriever):
            Creates and returns a compression retriever using the provided base retriever and reranker model.
        
        run_retriever():
            Runs the entire retrieval process by loading documents, creating embeddings, creating the retriever, 
            and applying compression.
    """
    
    def __init__(self, config: RetrieverConfig):
        """
        Initializes the Retriever instance with the given configuration.

        Args:
            config (RetrieverConfig): The configuration object for the retrieval process.
        """
        self.config = config

    def split_documents(self):
        """
        Loads and splits the documents from the specified file path into chunks.

        The documents are split using the `RecursiveCharacterTextSplitter` based on the chunk size and overlap
        defined in the configuration.

        Returns:
            list: A list of split documents.
        """
        loader = UnstructuredMarkdownLoader(self.config.file_path)
        loaded_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
        )
        docs = text_splitter.split_documents(loaded_documents)
        return docs

    def create_embeding_model(self):
        """
        Creates an embedding model using the configuration provided.

        Returns:
            embeddings: The embedding model used to convert documents into embeddings.
        """
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.config.embeding_model,
            model_kwargs={"device": DEVICE},
            encode_kwargs=encode_kwargs,
        )
        return embeddings

    @staticmethod
    def create__basic_retriever(docs, embeddings):
        """
        Creates a basic retriever using the provided documents and embeddings.

        Args:
            docs (list): The documents to be indexed.
            embeddings: The embedding model used for document indexing.

        Returns:
            retriever: A retriever object that can be used to search for relevant documents.
        """
        vectorstore_from_docs = FAISS.from_documents(docs, embedding=embeddings)
        retriever = vectorstore_from_docs.as_retriever(search_kwargs={"k": 10})
        return retriever

    def create_compression_retriever(self, retriever):
        """
        Creates a compression retriever by applying a reranker model on the base retriever.

        Args:
            retriever: The base retriever to be used.

        Returns:
            compression_retriever: A compression retriever that combines reranking and retrieval.
        """
        compressor = CohereRerank(
            model=self.config.reranker_model, cohere_api_key=self.config.COHERE_TOKEN
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        return compression_retriever

    def run_retriever(self):
        """
        Executes the entire retrieval process: splitting documents, creating embeddings, and generating the retriever.

        Returns:
            compression_retriever: The final compression retriever after applying reranking.
        """
        docs = self.split_documents()
        embeddings = self.create_embeding_model()
        retriever = self.create__basic_retriever(docs, embeddings)
        compression_retriever = self.create_compression_retriever(retriever)
        return compression_retriever

