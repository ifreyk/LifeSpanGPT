# LifeSpanGPT

## Overview
This project utilizes a Retrieval-Augmented Generation (RAG) pipeline to answer questions related to aging research on animals. It leverages the **bge-small-en** model for document embedding, Cohere **rerank-english-v3.0** for ranking the most relevant documents, and GPT-4 for generating answers to the queries. The system also computes metrics to evaluate the quality of the answers by comparing them to reference answers.

## Instalation
``` bash
git clone https://github.com/ifreyk/LifeSpanGPT.git
```
``` bash
pip install -r requirements.txt
```
## Usage
1) **tutorial.ipynb** - Notebook for quick run of the pipeline
2) **detailed_tutorial.ipynb** - Notebook with detailed description for each step of the pipeline
3) **metrics_calculation.ipynb** - Notebook for metric calculation
