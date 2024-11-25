from typing import List, Optional, Union

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


class Animal(BaseModel):
    species: str = Field(description="Species of the animal")
    strain: str = Field(description="Strain of the animal")
    group: str = Field(description="Control or experiment group")
    gender: str = Field(description="Sex of the animal")


class AnimalList(BaseModel):
    animals: list[Animal]

class Ntreatment(BaseModel):
    n_control: str = Field(
        description="Number of subjects in cotrol group relative to experiment group"
    )

class Ncontrol(BaseModel):
    n_treatment: str = Field(
        description="Number of subjects in experimental group"
    )

class Treatment(BaseModel):
    treatment: str = Field(
        description="What type of treatment or intervention are used"
    )


class WayOfAdministration(BaseModel):
    way_of_administration: str = Field(
        description="What way of administation are used"
    )


class AgeAtStart(BaseModel):
    age_at_start: int = Field(description="Age of the start of treamtment")


class DurationUnit(BaseModel):
    duration_unit: str = Field(
        description="In which units age of the start was Month/Week/Day and e.t.c"
    )


class Dosage(BaseModel):
    dosage: str = Field(description="Dosage of administration")


class MedianTreatment(BaseModel):
    median_treatment: Optional[float] = Field(
        description="Median treatment duration in units"
    )


class MaxTreatment(BaseModel):
    max_treatment: Optional[float] = Field(
        description="Max treatment duration in units"
    )


class TreatmentUnits(BaseModel):
    treatment_units: str = Field(description="In what units measured lifespan")


class PValue(BaseModel):
    p_value: Optional[str] = Field(description="p-value for statistical analysis")


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
            "treatment": Treatment,
            "way_of_administration": WayOfAdministration,
            "age_at_start": AgeAtStart,
            "duration_unit": DurationUnit,
            "dosage": Dosage,
            "median_treatment": MedianTreatment,
            "max_treatment": MaxTreatment,
            "treatment_units": TreatmentUnits,
            "p_value": PValue,
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
