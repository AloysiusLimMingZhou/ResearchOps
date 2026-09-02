import json
from retrieval.retriever import Retriever

def load_dataset(file_path:str) -> list[dict]: # Load json file as json has the format of [{}, {}, ...]
    with open(file_path, "r", encoding="utf-8") as file: # Open and read json from file_path
        return json.load(file)

def is_relevant(result, expected_filename:str, expected_pages:list[int]) -> bool: # Determine if the retrieved results matched with the atual results in terms of the same filename & page number searched
    return result.filename == expected_filename and result.page_number in expected_pages

def evaluate(retriever:Retriever, dataset_path:str):
    dataset = load_dataset(dataset_path)
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    reciprocal_rank_sum = 0

    for sample in dataset: # Loop through all sample question from json file
        results = retriever.retrieve(sample["question"], top_k=5) # Retrieve top 5 relevant chunks based on the sample question
        relevant_ranks = []

        for rank, result in enumerate(results, start=1):
            if is_relevant(result, sample["expected_filename"], sample["expected_pages"]): 
                relevant_ranks.append(rank)

            if any (rank <= 1 for rank in relevant_ranks):
                recall_at_1 += 1

            if any (rank <= 3 for rank in relevant_ranks):
                recall_at_3 += 1

            if any (rank <= 5 for rank in relevant_ranks):
                recall_at_5 += 1

            if relevant_ranks:
                first_rank = min(relevant_ranks)
                reciprocal_rank_sum += (1 / first_rank)

    total = len(dataset)
    print(f"Recall@1: {recall_at_1 / total:.3f}")
    print(f"Recall@3: {recall_at_3 / total:.3f}")
    print(f"Recall@5: {recall_at_5 / total:.3f}")
    print(f"MMR: {reciprocal_rank_sum / total:.3f}")