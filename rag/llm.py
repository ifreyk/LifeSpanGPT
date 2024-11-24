from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from typing import List, Union


class LLMConfig(BaseModel):
    model_name: Union[str, None] = Field(default=None)
    temperature: Union[float, None] = Field(default=None)
    api_key: Union[str, None] = Field(default=None)


class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config

    def create_llm(self):
        llm = ChatMistralAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            api_key=self.config.api_key,
        )
        return llm
