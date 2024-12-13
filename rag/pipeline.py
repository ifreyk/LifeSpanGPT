import os
import json
import time
from dotenv import load_dotenv

from rag.retriever import RetrieverConfig, Retriever
from rag.prompt_generator import PromptGeneratorConfig, PromptGenerator
from rag.llm import LLMConfig, LLM
from rag.qa import QA

load_dotenv()

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
        output = {"groups":[]}
        animal = PROMPT_CONFIG["animal"]
        query = animal["query"]
        prompt_config = PromptGeneratorConfig(
                prompt_intro=animal["prompt_intro"],
                prompt_base=animal["prompt_base"],
                prompt_type="animal"
        )
        prompt = self.run_prompt(prompt_config)
        answer = self.run_qa(
                    compression_retriever,
                    chat_llm,
                    prompt["parser"],
                    prompt["prompt"],
                    query,
        )
        output["groups"] = [i.dict() for i in answer.animals]
        for i,animal in enumerate(output["groups"]):
            animal_description = " ".join([value for key, value in animal.items() if value!=None])
            for key, value in PROMPT_CONFIG.items():
                if key == "animal":
                    continue
                else:
                    query = value["query"].format(animal=animal_description)
                    print(query)
                    prompt_config = PromptGeneratorConfig(
                        prompt_intro = value["prompt_intro"],
                        prompt_base = value["prompt_base"],
                        all_animals_description = animal_description,
                        prompt_type = key
                    )
                    prompt = self.run_prompt(prompt_config)
                    answer = self.run_qa(
                        compression_retriever,
                        chat_llm,
                        prompt["parser"],
                        prompt["prompt"],
                        query,
                    ).dict()
                    for key,value in answer.items():
                        for subject in value:
                            for sub_key,sub_value in subject.items():
                                output["groups"][i][sub_key] = sub_value
                    time.sleep(10)
        return output
