import os
import json
from dotenv import load_dotenv
import pandas as pd

from rag.retriever import RetrieverConfig, Retriever
from rag.prompt_generator import PromptGeneratorConfig, PromptGenerator
from rag.llm import LLMConfig, LLM
from rag.qa import QA

load_dotenv()

COHERE_TOKEN = os.getenv("COHERE_TOKEN")
MISTRAL_TOKEN = os.getenv("MISTRAL_TOKEN")
PROMPT_CONFIG_PATH = "config/prompts_config.json"


class LifeSpanGPT:
    def __init__(self, retriever_config: RetrieverConfig, llm_config: LLMConfig):
        self.retriever = Retriever(retriever_config)
        self.llm = LLM(llm_config)
        with open(PROMPT_CONFIG_PATH, "r", encoding="utf-8") as file:
            self.prompt_config = json.load(file)

    def run_retriever(self):
        print("Creating retriever")
        return self.retriever.run_retriever()

    def run_llm(self):
        print("Creating llm")
        return self.llm.create_llm()
    def get_animals(self,compression_retriever,chat_llm):
        value = self.prompt_config["animal"]
        query = value["query"]
        print(f"Detecting subjects...")
        prompt_config = PromptGeneratorConfig(
                    prompt_intro=value["prompt_intro"],
                    prompt_base=value["prompt_base"],
                    prompt_type="animal")
        prompt = self.run_prompt(prompt_config)
        answer = self.run_qa(
                    compression_retriever,
                    chat_llm,
                    prompt["parser"],
                    prompt["prompt"],
                    query)
        return answer
    @staticmethod
    def run_prompt(prompt_config: PromptGeneratorConfig):
        print("Generating prompt...")
        prompt_template = PromptGenerator(prompt_config)
        prompt = prompt_template.run_prompt()
        parser = prompt_template.create_parser()
        return {"prompt": prompt, "parser": parser}

    @staticmethod
    def run_qa(compression_retriever, chat_llm, parser, prompt, query):
        qa = QA(compression_retriever, chat_llm, parser, prompt)
        answer = qa.run_qa(query)
        return answer

    def run_pipeline(self):
        compression_retriever = self.run_retriever()
        chat_llm = self.run_llm()
        animals = self.get_animals(compression_retriever,chat_llm)
        result_df = pd.DataFrame(index=[x for x in range(len(animals.animals))])
        for i,animal in enumerate(animals.animals):
            animal_description = f"{animal.gender} {animal.species} {animal.group} {animal.strain}"
            for key, value in animal.dict().items():
                result_df.loc[i,key] = value
            return result_df
            for key, value in self.prompt_config.items():
                query = value["query"]
                print(f"Answering to question: {query}")
                if key == "animal":
                    continue
                else:
                    prompt_config = PromptGeneratorConfig(
                        prompt_intro=value["prompt_intro"],
                        prompt_base=value["prompt_base"],
                        all_animals_description=animal_description,
                        prompt_type=key,
                    )
                    prompt = self.run_prompt(prompt_config)
                    answer = self.run_qa(
                        compression_retriever,
                        chat_llm,
                        prompt["parser"],
                        prompt["prompt"],
                        query,
                    )
                    for field_name, _ in answer.model_fields.items():
                        result_df.loc[i,field_name] = answer.field_name
        return result_df
