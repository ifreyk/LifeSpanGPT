# LifeSpanGPT

## Overview
This project utilizes a Retrieval-Augmented Generation (RAG) pipeline to answer questions related to aging research on animals. It leverages the **bge-small-en** model for document embedding, Cohere **rerank-english-v3.0** for ranking the most relevant documents, and GPT-4 for generating answers to the queries. The system also computes metrics to evaluate the quality of the answers by comparing them to reference answers.

## Repository structure
- **Root Directory**:
  - **`README.md`** - The main documentation file where users can find project information and setup instructions.
  - **`requirements.txt`** - Contains all the dependencies required to run the project.
  - **`tutorial.ipynb`** - Notebook for quick run of the pipeline
  - **`detailed_tutorial.ipynb`** - Notebook with detailed description for each step of the pipeline
  - **`metrics_calculation.ipynb`** - Notebook for metric calculation
- **config**:
  - **`prompts_config`.json** - JSON config with paths to prompts
- **rag**:
  - **`llm.py`** - Class for creating LLM object
  - **`prompt_generator.py`** - Class for creating prompts with exact format
  - **`retriever.py`** - Class for creating retriever
  - **`qa.py`** - Class for combining llm,prompt and retriever
  - **`pipeline.py`** - Full pipeline
- **templates**:
  - **prompt_bases**:
    - **`animal_details.txt`** - File with body of the prompt to get information about animal treatment
    - **`animal_results.txt`** - File with body of the prompt to get information about animal survival results
    - **`animal.txt`** - File with body of the prompt to get information about experimantal groups in the article
  - **prompt_intro**:
    - **`animal_details.txt`** - File with intro to the prompt body to get information about animal treatment
    - **`animal_results.txt`** - File with intro to the prompt body to get information about animal survival results
    - **`animal.txt`** - File with intro to the prompt body to get information about experimantal groups in the article
- **utils.py**:
  - **`parser.py`** - Class for parsing .pdf files to .md files for the pipeline
## Instalation
``` bash
git clone https://github.com/ifreyk/LifeSpanGPT.git
```
``` bash
pip install -r requirements.txt
```
## Usage
1) **`tutorial.ipynb`** - Notebook for quick run of the pipeline
2) **`detailed_tutorial.ipynb`** - Notebook with detailed description for each step of the pipeline
3) **`metrics_calculation.ipynb`** - Notebook for metric calculation
