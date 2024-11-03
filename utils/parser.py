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
    path_to_file: str
    llama_cloud_token: str
    instruction: Union[str, None] = Field(default=None)


class Parser:
    def __init__(self, config: ParserConfig):
        self.config = config
        self.file_name = pathlib.Path(self.config.path_to_file).stem + ".md"
        if not os.path.exists(PROCESSED_DATA_ROOT):
            os.makedirs(PROCESSED_DATA_ROOT, exist_ok=True)

    def create_parser(self):
        self.parser = LlamaParse(
            api_key=self.config.llama_cloud_token,
            result_type="markdown",
            parsing_instruction=self.config.instruction,
            max_timeout=5000,
        )

    def parse(self):
        llama_parse_documents = self.parser.load_data(self.config.path_to_file)
        full_text = "".join([x.text for x in llama_parse_documents])
        document_path = pathlib.Path(os.path.join(PROCESSED_DATA_ROOT, self.file_name))
        with document_path.open("a") as f:
            f.write(full_text)
