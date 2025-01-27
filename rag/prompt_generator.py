from typing import List, Optional, Union

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


class Animal(BaseModel):
    """
    Represents the basic details of an animal used in an experiment or study.

    Attributes:
        species (Union[str, None]): The species of the animal (e.g., "Mouse", "Rat").
        strain (Union[str, None]): The strain of the animal, if applicable.
        group (Union[str, None]): The name of the subject group (e.g., "Treatment Group").
        gender (Union[str, None]): The sex of the animal (e.g., "Male", "Female").
    """
    species: Union[str, None] = Field(description="Species of the animal")
    strain: Union[str, None] = Field(description="Strain of the animal")
    group: Union[str, None] = Field(description="Name of the subject group")
    gender: Union[str, None] = Field(description="Sex of the animal")



class AnimalList(BaseModel):
    """
    A container for a list of `Animal` objects.

    Attributes:
        animals (list[Animal]): A list of `Animal` instances that belong to the same experiment or study group.
    """
    animals: list[Animal]


class AnimalDetails(BaseModel):
    """
    Contains detailed information about the treatment, age, and dosage for each animal.

    Attributes:
        treatment (Union[str, None]): The type of treatment or intervention used for the animal.
        way_of_administration (Union[str, None]): The method by which the treatment was administered.
        age_at_start (Union[int, str, None]): The age of the animal when treatment started.
        duration_unit (Union[str, None]): The unit for the `age_at_start` (e.g., "Months", "Weeks").
        dosage (Union[str, int, None]): The dosage of the treatment administered.
    """
    treatment: Union[str, None] = Field(
        description="What type of treatment or intervention are used?"
    )
    way_of_administration: Union[str, None] = Field(
        description="What way of administration are used?"
    )
    age_at_start: Union[int, str, None] = Field(description="Age of the start of treatment")
    duration_unit: Union[str, None] = Field(
        description="In which units age of the start was Month/Week/Day and e.t.c"
    )
    dosage: Union[str, int, None] = Field(description="Dosage of administration")



class AnimalDetailsList(BaseModel):
    """
    A container for a list of `AnimalDetails` objects.

    Attributes:
        animal_details (List[AnimalDetails]): A list of detailed treatment and administration information for animals.
    """
    animal_details: List[AnimalDetails]



class AnimalResults(BaseModel):
    """
    Represents the results of the treatment in the study, including statistical data.

    Attributes:
        n_treatment (Union[int, str, None]): The number of animals in the treatment group.
        n_control (Union[int, str, None]): The number of animals in the control group.
        median_treatment (Union[int, float, str, None]): The median treatment duration.
        max_treatment (Union[int, float, str, None]): The maximum treatment duration.
        treatment_units (Union[str, None]): The units in which the treatment duration is measured.
        p_value (Union[str, float, None]): The p-value for statistical significance.
        median_control (Union[int, float, str, None]): The median duration in the control group.
        max_control (Union[int, float, str, None]): The maximum duration in the control group.
    """
    n_treatment: Union[int, str, None] = Field(description="Number of animals in this group")
    n_control: Union[int, str, None] = Field(description="Number of animals in control group")
    median_treatment: Union[int, float, str, None] = Field(
        description="Median treatment duration in units"
    )
    max_treatment: Union[int, float, str, None] = Field(
        description="Max treatment duration in units"
    )
    treatment_units: Union[str, None] = Field(description="In what units measured lifespan")
    p_value: Union[str, float, None] = Field(description="p-value for statistical analysis")
    median_control: Union[int, float, str, None] = Field(
        description="Median treatment duration in units"
    )
    max_control: Union[int, float, str, None] = Field(
        description="Max treatment duration in units"
    )


class AnimalResultsList(BaseModel):
    """
    A container for a list of `AnimalResults` objects.

    Attributes:
        animal_results (List[AnimalResults]): A list of the results for different animals or groups.
    """
    animal_results: List[AnimalResults]



class PromptGeneratorConfig(BaseModel):
    """
    Configuration model for generating prompts for different types of animal-related data.

    Attributes:
        prompt_intro (Union[str, None]): Path to the introductory part of the prompt.
        prompt_base (Union[str, None]): Path to the base part of the prompt.
        all_animals_description (Union[str, None]): Description of all animals involved in the study.
        prompt_type (Union[str, None]): The type of prompt being generated, e.g., "animal", "animal_details", "animal_results".
    """
    prompt_intro: Union[str, None] = Field(default=None)
    prompt_base: Union[str, None] = Field(default=None)
    all_animals_description: Union[str, None] = Field(default=None)
    prompt_type: Union[str, None] = Field(
        default=None
    )  # animal, animal_details, animal_results



class PromptGenerator:
    """
    A class for generating prompts based on the provided configuration and animal-related data.

    Attributes:
        config (PromptGeneratorConfig): The configuration that defines how prompts should be generated.
        output_class (dict): A mapping between prompt types and corresponding Pydantic models.

    Methods:
        __init__(config: PromptGeneratorConfig):
            Initializes the `PromptGenerator` with the provided configuration.

        create_parser():
            Creates a parser based on the prompt type defined in the configuration.

        load_prompt_intro():
            Loads the introductory part of the prompt from a file.

        load_prompt_base():
            Loads the base part of the prompt from a file.

        create_full_prompt(prompt_intro_file, prompt_base_file):
            Combines the intro and base files to generate a complete prompt template.

        run_prompt():
            Generates the full prompt and returns it.
    """
    
    def __init__(self, config: PromptGeneratorConfig):
        """
        Initializes the `PromptGenerator` with the provided configuration.

        Args:
            config (PromptGeneratorConfig): The configuration for generating prompts.
        """
        self.config = config
        self.output_class = {
            "animal": AnimalList,
            "animal_details": AnimalDetailsList,
            "animal_results": AnimalResultsList,
        }

    def create_parser(self):
        """
        Creates a parser for the specified prompt type.

        Returns:
            PydanticOutputParser: The parser to process the corresponding Pydantic model.
        """
        choose_class = self.output_class[self.config.prompt_type]
        return PydanticOutputParser(pydantic_object=choose_class)

    def load_prompt_intro(self):
        """
        Loads the introductory part of the prompt from the file specified in the configuration.

        Returns:
            str: The content of the introductory prompt file.
        """
        with open(self.config.prompt_intro, "r", encoding="utf-8") as file:
            prompt_intro_file = file.read()
        return prompt_intro_file.format(
            all_animals_description=self.config.all_animals_description
        )

    def load_prompt_base(self):
        """
        Loads the base part of the prompt from the file specified in the configuration.

        Returns:
            str: The content of the base prompt file.
        """
        with open(self.config.prompt_base, "r", encoding="utf-8") as file:
            prompt_base_file = file.read()
        return prompt_base_file

    def create_full_prompt(self, prompt_intro_file, prompt_base_file):
        """
        Combines the introductory and base parts of the prompt and creates a `PromptTemplate`.

        Args:
            prompt_intro_file (str): The introductory content of the prompt.
            prompt_base_file (str): The base content of the prompt.

        Returns:
            PromptTemplate: The final prompt template for use in the prompt generation process.
        """
        parser = self.create_parser()
        prompt_template = prompt_intro_file + prompt_base_file
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query", "context"],
            partial_variables={"format_instructions": parser.model_json_schema()},
        )
        return prompt

    def run_prompt(self):
        """
        Generates and returns the full prompt by loading the intro and base files, and combining them.

        Returns:
            PromptTemplate: The complete prompt template ready for use.
        """
        prompt_intro_file = self.load_prompt_intro()
        prompt_base_file = self.load_prompt_base()
        prompt = self.create_full_prompt(prompt_intro_file, prompt_base_file)
        return prompt
