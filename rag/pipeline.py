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
    """
    LifeSpanGPT is a class responsible for running a pipeline that retrieves, processes,
    and analyzes data related to lifespan research using various components like retrieval
    models, LLMs (Large Language Models), and prompt generation.
    """

    def __init__(self, retriever_config: RetrieverConfig, llm_config: LLMConfig):
        """
        Initializes the LifeSpanGPT class with the provided configuration for the retriever and LLM.

        Args:
            retriever_config (RetrieverConfig): Configuration for the document retrieval process.
            llm_config (LLMConfig): Configuration for the LLM (Large Language Model) for answering queries.
        """
        self.retriever = Retriever(retriever_config)
        self.llm = LLM(llm_config)

    def run_retriever(self):
        """
        Creates and runs the retriever to fetch relevant documents for processing.

        Returns:
            compression_retriever: The retriever that will be used to fetch and compress documents.
        """
        print("Creating retriever")
        return self.retriever.run_retriever()

    def run_llm(self):
        """
        Creates and initializes the LLM (Large Language Model) for answering queries.

        Returns:
            chat_llm: The LLM that will be used for answering queries.
        """
        print("Creating llm")
        return self.llm.create_llm()

    @staticmethod
    def run_prompt(prompt_config: PromptGeneratorConfig):
        """
        Generates the prompt based on the provided configuration for different research tasks.

        Args:
            prompt_config (PromptGeneratorConfig): The configuration for generating the prompt.

        Returns:
            dict: A dictionary containing the generated prompt and the corresponding parser.
        """
        print("Generating prompt...")
        prompt_template = PromptGenerator(prompt_config)
        prompt = prompt_template.run_prompt()
        parser = prompt_template.create_parser()
        return {"prompt": prompt, "parser": parser}

    @staticmethod
    def run_qa(compression_retriever, chat_llm, parser, prompt, query):
        """
        Runs a QA pipeline to answer a query using the retriever, LLM, and prompt.

        Args:
            compression_retriever: The retriever responsible for fetching relevant documents.
            chat_llm: The LLM for generating answers based on the documents and query.
            parser: The parser used to process the answer.
            prompt: The prompt used to query the LLM.
            query (str): The query to be answered.

        Returns:
            answer: The processed answer after the QA pipeline.
        """
        qa = QA(compression_retriever, chat_llm, parser, prompt)
        answer = qa.run_qa(query)
        return answer

    def run_pipeline(self):
        """
        Runs the entire pipeline to retrieve relevant documents, process them, generate prompts,
        and answer queries based on lifespan-related data.

        This method calls the retrieval, LLM, and QA processes iteratively to generate a comprehensive
        output, and it formats the output into a structured format.

        Returns:
            dict: A dictionary containing the results of the processed pipeline, including data on
            animal groups and additional information gathered from the QA results.
        """
        compression_retriever = self.run_retriever()
        chat_llm = self.run_llm()
        output = {"groups": []}
        animal = PROMPT_CONFIG["animal"]
        query = animal["query"]
        prompt_config = PromptGeneratorConfig(
            prompt_intro=animal["prompt_intro"],
            prompt_base=animal["prompt_base"],
            prompt_type="animal",
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
        for i, animal in enumerate(output["groups"]):
            animal_description = " ".join(
                [value for key, value in animal.items() if value != None]
            )
            for key, value in PROMPT_CONFIG.items():
                if key == "animal":
                    continue
                else:
                    query = value["query"].format(animal=animal_description)
                    print(query)
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
                    ).dict()
                    for key, value in answer.items():
                        for subject in value:
                            for sub_key, sub_value in subject.items():
                                output["groups"][i][sub_key] = sub_value
                    time.sleep(10)
        return output
