# %%
import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
import time

from rag.retriever import RetrieverConfig
from rag.llm import LLMConfig
from rag.pipeline import LifeSpanGPT

load_dotenv()
COHERE_TOKEN = os.getenv("COHERE_TOKEN")
MISTRAL_TOKEN = os.getenv("MISTRAL_TOKEN")
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
# %%
for file in tqdm(os.listdir("processed_data")):
    file_name = file.split(".md")[0]
    if file_name+".json" not in os.listdir("pipeline_results/openai"):
        print(file_name)
        try:
            config = RetrieverConfig(
                file_path=f"processed_data/{file}",
                embeding_model="BAAI/bge-small-en",
                reranker_model="rerank-english-v3.0",
                chunk_size=20000,
                chunk_overlap=2000,
                COHERE_TOKEN=COHERE_TOKEN,
            )
            llm_config = LLMConfig(
                model_name="gpt-4o", temperature=0.0, api_key=OPENAI_TOKEN
            )
            pipeline = LifeSpanGPT(config, llm_config)
            answer = pipeline.run_pipeline()
            for key,value in answer.items():
                answer[key] = value.dict()
            with open(f"pipeline_results/openai/{file_name}.json","w",encoding="utf-8") as f:
                json.dump(answer, f, indent=4)
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
        time.sleep(30)
# %%
for key,value in answer.items():
    break
# %%
