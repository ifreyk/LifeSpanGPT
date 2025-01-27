from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from typing import List, Union
from langchain_openai import ChatOpenAI

class LLMConfig(BaseModel):
    """
    Configuration model for setting up the language model (LLM).

    Attributes:
        model_name (Union[str, None]): The name of the language model to use (e.g., "gpt-3.5-turbo"). Default is None.
        temperature (Union[float, None]): Controls randomness in the model's responses. A value between 0 and 1. Default is None.
        api_key (Union[str, None]): The API key for authenticating with the language model API. Default is None.

    Example:
        config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7, api_key="your_api_key_here")
    """
    model_name: Union[str, None] = Field(default=None)
    temperature: Union[float, None] = Field(default=None)
    api_key: Union[str, None] = Field(default=None)


class LLM:
    """
    A class to interface with a language model, such as OpenAI's GPT.

    Attributes:
        config (LLMConfig): Configuration settings for the language model, including model name, temperature, and API key.

    Methods:
        __init__(config: LLMConfig):
            Initializes the LLM instance with the provided configuration.
        
        create_llm():
            Creates and returns an instance of the language model (ChatOpenAI) using the configuration provided.
    
    Example:
        config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7, api_key="your_api_key_here")
        llm_instance = LLM(config)
        language_model = llm_instance.create_llm()
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initializes the LLM instance with the provided configuration.

        Args:
            config (LLMConfig): Configuration containing the model name, temperature, and API key.
        """
        self.config = config

    def create_llm(self):
        """
        Creates an instance of the language model (ChatOpenAI) using the provided configuration.

        Returns:
            ChatOpenAI: The language model instance configured with the specified settings.
        
        Example:
            llm = LLM(config)
            model = llm.create_llm()
        """
        llm = ChatOpenAI(
            name=self.config.model_name,
            temperature=self.config.temperature,
            api_key=self.config.api_key,
        )
        return llm

