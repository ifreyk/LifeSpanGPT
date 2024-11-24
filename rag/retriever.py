from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.vectorstores import FAISS
from langchain.retrievers.document_compressors import CohereRerank
from pydantic import BaseModel, Field
from typing import List, Union


class RetrieverConfig(BaseModel):
    file_path: Union[str, None] = Field(default=None)
    embeding_model: Union[str, None] = Field(default=None)
    reranker_model: Union[str, None] = Field(default=None)
    chunk_size: Union[int, None] = Field(default=None)
    chunk_overlap: Union[int, None] = Field(default=None)
    COHERE_TOKEN: Union[str, None] = Field(default=None)


class Retriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config

    def split_documents(self):
        loader = UnstructuredMarkdownLoader(self.config.file_path)
        loaded_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
        )
        docs = text_splitter.split_documents(loaded_documents)
        return docs

    def create_embeding_model(self):
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.config.embeding_model,
            model_kwargs={"device": "mps"},
            encode_kwargs=encode_kwargs,
        )
        return embeddings

    @staticmethod
    def create__basic_retriever(docs, embeddings):
        vectorstore_from_docs = FAISS.from_documents(docs, embedding=embeddings)
        retriever = vectorstore_from_docs.as_retriever(search_kwargs={"k": 10})
        return retriever

    def create_compression_retriever(self, retriever):
        compressor = CohereRerank(
            model=self.config.reranker_model, cohere_api_key=self.config.COHERE_TOKEN
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        return compression_retriever

    def run_retriever(self):
        docs = self.split_documents()
        embeddings = self.create_embeding_model()
        retriever = self.create__basic_retriever(docs, embeddings)
        compression_retriever = self.create_compression_retriever(retriever)
        return compression_retriever
