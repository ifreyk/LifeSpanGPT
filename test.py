# %%
import os
from dotenv import load_dotenv

from rag.retriever import RetrieverConfig
from rag.llm import LLMConfig
from rag.pipeline import LifeSpanGPT

load_dotenv()
COHERE_TOKEN = os.getenv("COHERE_TOKEN")
MISTRAL_TOKEN = os.getenv("MISTRAL_TOKEN")
# %%
from rag.pipeline import LifeSpanGPT

config = RetrieverConfig(
    file_path="processed_data/s41418-019-0422-6.md",
    embeding_model="BAAI/bge-small-en",
    reranker_model="rerank-english-v3.0",
    chunk_size=15000,
    chunk_overlap=2000,
    COHERE_TOKEN=COHERE_TOKEN,
)
llm_config = LLMConfig(
    model_name="mistral-large-latest", temperature=0.0, api_key=MISTRAL_TOKEN
)
# %%
pipeline = LifeSpanGPT(config, llm_config)
# %%
answer = pipeline.run_pipeline()