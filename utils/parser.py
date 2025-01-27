import os
from dotenv import load_dotenv
import pathlib
import nest_asyncio

from pydantic import BaseModel, Field
from typing import Union
from llama_parse import LlamaParse

nest_asyncio.apply()
load_dotenv()

PROCESSED_DATA_ROOT = "processed_data/"


class ParserConfig(BaseModel):
    """
    Configuration model for setting up the parser.

    Attributes:
        path_to_file (str): The file path to the document that will be parsed.
        llama_cloud_token (str): LlamaParse API key used to authenticate requests.
        instruction (Union[str, None]): Optional instruction to guide the parsing process. Defaults to None.

    Example:
        config = ParserConfig(path_to_file="input.txt", llama_cloud_token="your_api_key_here")
    """
    path_to_file: str
    llama_cloud_token: str
    instruction: Union[str, None] = Field(default=None)


class Parser:
    """
    A class that processes a document using the LlamaParse API and saves the output as a markdown file.

    Attributes:
        config (ParserConfig): Configuration settings, including file path and LlamaParse API key.
        file_name (str): The name of the output markdown file based on the input file's name.

    Methods:
        __init__(config: ParserConfig):
            Initializes the parser instance with the provided configuration.
        
        create_parser():
            Initializes the LlamaParse parser with the provided API key and parsing instruction.
        
        parse():
            Loads the input document, processes the content using the LlamaParse API, 
            concatenates the results, and saves them as a markdown file in the `processed_data/` directory.
    
    Example:
        config = ParserConfig(path_to_file="input.txt", llama_cloud_token="your_api_key_here")
        parser = Parser(config)
        parser.create_parser()
        parser.parse()
    """
    
    def __init__(self, config: ParserConfig):
        """
        Initializes the Parser instance with the provided configuration.

        Args:
            config (ParserConfig): Configuration for the parser, containing the file path and LlamaParse API key.

        Creates the necessary directories for saving the processed document if they do not already exist.
        """
        self.config = config
        self.file_name = pathlib.Path(self.config.path_to_file).stem + ".md"
        if not os.path.exists(PROCESSED_DATA_ROOT):
            os.makedirs(PROCESSED_DATA_ROOT, exist_ok=True)

    def create_parser(self):
        """
        Initializes the LlamaParse parser with the provided API key and parsing instructions.

        Configures the parser to generate markdown output and sets the maximum timeout for requests.
        """
        self.parser = LlamaParse(
            api_key=self.config.llama_cloud_token,
            result_type="markdown",
            parsing_instruction=self.config.instruction,
            max_timeout=5000,
        )

    def parse(self):
        """
        Loads and processes the input document using the LlamaParse API.

        Combines the parsed content from multiple documents into a single string and writes the result
        as a markdown file in the `processed_data/` directory.

        Raises:
            FileNotFoundError: If the input file specified in `path_to_file` is not found.
            LlamaParseError: If there's an error processing the document with the LlamaParse API.
        """
        llama_parse_documents = self.parser.load_data(self.config.path_to_file)
        full_text = "".join([x.text for x in llama_parse_documents])
        document_path = pathlib.Path(os.path.join(PROCESSED_DATA_ROOT, self.file_name))
        with document_path.open("a") as f:
            f.write(full_text)
