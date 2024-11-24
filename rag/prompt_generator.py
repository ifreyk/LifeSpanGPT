from typing import List, Optional, Union

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


class Animal(BaseModel):
    species: str = Field(description="Species of the animal")
    strain: str = Field(description="Strain of the animal")
    group: str = Field(description="Control or experiment group")
    gender: str = Field(description="Sex of the animal")
    n_treatment: Optional[int] = Field(description="Number of animals in this group")
    n_control: Optional[int] = Field(description="Number of animals in control group")


class AnimalList(BaseModel):
    animals: list[Animal]


class AnimalDetails(BaseModel):
    treatment: str = Field(
        description="What type of treatment or intervention are used?"
    )
    way_of_administration: str = Field(
        description="What way of administation are used?"
    )
    age_at_start: int = Field(description="Age of the start of treamtment")
    duration_unit: str = Field(
        description="In which units age of the start was Month/Week/Day and e.t.c"
    )
    dosage: str = Field(description="Dosage of administration")


class AnimalDetailsList(BaseModel):
    animal_details: List[AnimalDetails]


class AnimalResults(BaseModel):
    median_treatment: Optional[float] = Field(
        description="Median treatment duration in units"
    )
    max_treatment: Optional[float] = Field(
        description="Max treatment duration in units"
    )
    treatment_units: str = Field(description="In what units measured lifespan")
    p_value: Optional[str] = Field(description="p-value for statistical analysis")


class AnimalResultsList(BaseModel):
    animal_results: List[AnimalResults]


class PromptGeneratorConfig(BaseModel):
    prompt_intro: Union[str, None] = Field(default=None)
    prompt_base: Union[str, None] = Field(default=None)
    all_animals_description: Union[str, None] = Field(default=None)
    prompt_type: Union[str, None] = Field(
        default=None
    )  # animal, animal_details,animal_result


class PromptGenerator:
    def __init__(self, config: PromptGeneratorConfig):
        self.config = config
        self.output_class = {
            "animal": AnimalList,
            "animal_details": AnimalDetailsList,
            "animal_results": AnimalResultsList,
        }

    def create_parser(self):
        choose_class = self.output_class[self.config.prompt_type]
        return PydanticOutputParser(pydantic_object=choose_class)

    def load_prompt_intro(self):
        with open(self.config.prompt_intro, "r", encoding="utf-8") as file:
            prompt_intro_file = file.read()
        return prompt_intro_file.format(
            all_animals_description=self.config.all_animals_description
        )

    def load_prompt_base(self):
        with open(self.config.prompt_base, "r", encoding="utf-8") as file:
            prompt_base_file = file.read()
        return prompt_base_file

    def craete_full_prompt(self, prompt_intro_file, prompt_base_file):
        parser = self.create_parser()
        prompt_template = prompt_intro_file + prompt_base_file
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query", "context"],
            partial_variables={"format_instructions": parser.model_json_schema()},
        )
        return prompt

    def run_prompt(self):
        prompt_intro_file = self.load_prompt_intro()
        prompt_base_file = self.load_prompt_base()
        prompt = self.craete_full_prompt(prompt_intro_file, prompt_base_file)
        return prompt