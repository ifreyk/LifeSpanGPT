import os
import json
from dotenv import load_dotenv

from rag.retriever import RetrieverConfig, Retriever
from rag.prompt_generator import PromptGeneratorConfig, PromptGenerator
from rag.llm import LLMConfig, LLM
from rag.qa import QA

load_dotenv()

COHERE_TOKEN = os.getenv("COHERE_TOKEN")
MISTRAL_TOKEN = os.getenv("MISTRAL_TOKEN")
PROMPT_CONFIG_PATH = "config/prompts_config.json"

with open(PROMPT_CONFIG_PATH, "r", encoding="utf-8") as file:
    PROMPT_CONFIG = json.load(file)


class LifeSpanGPT:
    def __init__(self, retriever_config: RetrieverConfig, llm_config: LLMConfig):
        self.retriever = Retriever(retriever_config)
        self.llm = LLM(llm_config)

    def run_retriever(self):
        print("Creating retriever")
        return self.retriever.run_retriever()

    def run_llm(self):
        print("Creating llm")
        return self.llm.create_llm()

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
        output = {}
        for key, value in PROMPT_CONFIG.items():
            query = value["query"]
            print(f"Answering to question: {query}")
            if key == "animal":
                prompt_config = PromptGeneratorConfig(
                    prompt_intro=value["prompt_intro"],
                    prompt_base=value["prompt_base"],
                    prompt_type=key
                )
                prompt = self.run_prompt(prompt_config)
                answer = self.run_qa(
                    compression_retriever,
                    chat_llm,
                    prompt["parser"],
                    prompt["prompt"],
                    query,
                )
                output[key] = answer
                animal_descriptions = [
                f"{animal.gender} {animal.species} {animal.group} {animal.strain}" 
                for animal in answer.animals]
                all_animals_description = ", ".join(animal_descriptions)
            else:
                prompt_config = PromptGeneratorConfig(
                    prompt_intro = value["prompt_intro"],
                    prompt_base = value["prompt_base"],
                    all_animals_description = all_animals_description,
                    prompt_type = key
                )
                prompt = self.run_prompt(prompt_config)
                answer = self.run_qa(
                    compression_retriever,
                    chat_llm,
                    prompt["parser"],
                    prompt["prompt"],
                    query,
                )
                output[key] = answer
        return output
