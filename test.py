# %%
import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


from rag.retriever import RetrieverConfig
from rag.llm import LLMConfig
from rag.pipeline import LifeSpanGPT

load_dotenv()
COHERE_TOKEN = os.getenv("COHERE_TOKEN_DIMA")
MISTRAL_TOKEN = os.getenv("MISTRAL_TOKEN")
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
# %%
for file in tqdm(os.listdir("processed_data")):
    file_name = file.split(".md")[0]
    if file_name + ".json" not in os.listdir("pipeline_results/openai"):
        print(file_name)
        config = RetrieverConfig(
            file_path=f"processed_data/{file}",
            embeding_model="BAAI/bge-small-en",
            reranker_model="rerank-english-v3.0",
            chunk_size=15000,
            chunk_overlap=2000,
            COHERE_TOKEN=COHERE_TOKEN,
        )
        llm_config = LLMConfig(
            model_name="gpt-4o", temperature=0.0, api_key=OPENAI_TOKEN
        )
        pipeline = LifeSpanGPT(config, llm_config)
        answer = pipeline.run_pipeline()
        with open(
            f"pipeline_results/openai/{file_name}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(answer, f, indent=4)
        time.sleep(60)
# %%
result_df = pd.DataFrame()
# %%
for file in os.listdir("pipeline_results/openai"):
    if not file.__contains__(".DS"):
        file_path = os.path.join("pipeline_results/openai", file)
        with open(file_path, "r") as f:
            data = json.load(f)
    temp_df = pd.DataFrame(index=range(len(data["groups"])))
    temp_df["doi"] = file.split(".json")[0]
    for i, j in enumerate(data["groups"]):
        for key, value in j.items():
            temp_df.loc[i, key] = value
            temp_df.loc[i, key] = value
            temp_df.loc[i, key] = value
            temp_df.loc[i, key] = value
    temp_df = temp_df.replace("null", None)
    temp_df = temp_df.replace("null", None)
    result_df = pd.concat([result_df, temp_df], ignore_index=True)
# %%
result_df.to_excel("lifespangpt_results_v2.xlsx", index=False)
# %%
reference = pd.read_excel("CollidaData_2023.xlsx")
reference = reference.rename(columns={"intervention": "treatment"})
test = pd.read_excel("lifespangpt_results_v2.xlsx")
model = SentenceTransformer("all-mpnet-base-v2")

# %%
fields = [
    "species",
    "strain",
    "gender",
    "treatment",
    "way_of_administration",
    "dosage",
    "age_at_start",
    "duration_unit",
    "median_treatment",
    "max_treatment",
    "median_control",
    "max_control",
    "p_value",
    "n_treatment",
    "n_control",
]
# %%
full_scores = []
for i in tqdm(os.listdir("pipeline_results/reference_data")):
    reference_path = os.path.join("pipeline_results/reference_data", i)
    answer_path = os.path.join("pipeline_results/openai", i)
    reference_answer = json.load(open(reference_path, "r"))
    answer_answer = json.load(open(answer_path, "r"))
    scores = []
    reference_len = len(reference_answer["groups"])
    answer_len = len(answer_answer["groups"])
    # Calculate the similarity metric
    group_similarity = 1 / (1 + abs(answer_len - reference_len) / reference_len)
    for answer in reference_answer["groups"]:
        ref = " ".join([str(answer[fld]) for fld in fields])
        ref_embed = model.encode(ref)
        groups = answer_answer["groups"]
        answers = []
        for res in groups:
            res_temp = [str(res[fld]) for fld in fields]
            answers.append(" ".join(res_temp))
        res_embed = model.encode(answers)
        test = max([cosine_similarity([ref_embed], [x])[0][0] for x in res_embed])
        scores.append(test)
        # simmilaritys = [cosine_similarity(query,x) for x in embeddings]
    metric = np.mean(scores)*group_similarity
    print(metric)
    full_scores.append(metric)
#%%
full_scores_random = []
for i in tqdm(os.listdir("pipeline_results/reference_data")):
    file = random.choice([x for x in os.listdir("pipeline_results/openai") if x!='.DS_Store'])
    reference_path = os.path.join("pipeline_results/reference_data", i)
    answer_path = os.path.join("pipeline_results/openai", file)
    reference_answer = json.load(open(reference_path, "r"))
    answer_answer = json.load(open(answer_path, "r"))
    scores = []
    reference_len = len(reference_answer["groups"])
    answer_len = len(answer_answer["groups"])
    # Calculate the similarity metric
    group_similarity = 1 / (1 + abs(answer_len - reference_len) / reference_len)
    for answer in reference_answer["groups"]:
        ref = " ".join([str(answer[fld]) for fld in fields])
        ref_embed = model.encode(ref)
        groups = answer_answer["groups"]
        answers = []
        for res in groups:
            res_temp = [str(res[fld]) for fld in fields]
            answers.append(" ".join(res_temp))
        res_embed = model.encode(answers)
        test = max([cosine_similarity([ref_embed], [x])[0][0] for x in res_embed])
        scores.append(test)
        # simmilaritys = [cosine_similarity(query,x) for x in embeddings]
    metric = np.mean(scores)*group_similarity
    print(metric)
    full_scores_random.append(metric)
# %%
sns.displot(full_scores)
plt.title("Cosine Similarity Distribution (Normal Selection)")
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Article count")
# %%
random.choice(os.listdir("pipeline_results/openai"))
# %%
stat, p_value = mannwhitneyu(full_scores_random, full_scores, alternative='two-sided')  # 'two-sided', 'greater', or 'less'

print(f"U-statistic: {stat}")
print(f"P-value: {p_value}")
