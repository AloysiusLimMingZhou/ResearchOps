import json
from retrieval.retriever import Retriever

K_VALUES = [1, 3, 5]

def load_dataset(file_path:str) -> list[dict]: # Load json file as json has the format of [{}, {}, ...]
    with open(file_path, "r", encoding="utf-8") as file: # Open and read json from file_path
        return json.load(file)

def is_relevant(result, expected_filename:str, expected_pages:list[int]) -> bool: # Determine if the retrieved results matched with the atual results in terms of the same filename & page number searched
    return result.filename == expected_filename and result.page_number in expected_pages

def evaluate(retriever:Retriever, dataset_path:str):
    dataset = load_dataset(dataset_path) # Load Vector DB
    hit_sums = {k: 0 for k in K_VALUES} # Create a dict with k = 0 for hit@1, hit@3, hit@5 (i.e. {hit@1: 0, hit@3: 0, hit@5: 0}). If expected chunks hit top 1 retrieved result, hit@1 += 1. 
                                        # hit@k calculates if there's relevant chunks in top k retrieved chunks. (i.e. Query 1 has relevant chunk in 1st retrieved chunk, hit@1 = 1, else hit@1 = 0)
    
    precision_sums = {k: 0.0 for k in K_VALUES} # Same as hit_sums but for precision@1, precision@3, precision@5. 
                                                # precision@k is calculate percentage of relevant chunks among all k chunks retrieved. (relevant_chunks / k). 
                                                # (i.e. top 5 chunks only 2 chunks are relevant based on filename & page number, then 2/5 = 40% Precision@5)
    
    page_recall_sums = {k: 0.0 for k in K_VALUES} # Same as hit_sums but for page_recall@1, page_recall@3, page_recall@5. 
                                                  # page_recall@k is calculate percentage of relevant retrieved chunks among all possible relevant chunks (top_k_relevant_chunks / total_relevant_chunks). (i.e. relevant_pages = {4, 5, 6}, top_k_chunks = {4, 5, 4, 5, 2}, among 5 chunks, 2 unique ones hit. So 2/3 = 67% PagedRecall@5)
                                                  # We use page_recall@k here instead of chunk_recall@k for simplicity. Page recall compare relevant pages of filenames instead of specific relevant chunks

    reciprocal_rank_sum = 0 # Mean Reciprocal Rank. Basically take multiple queries and retrieve chunks, compare retrieved chunks if it matches with the expected filename & page. 
                            # If the relevant chunk is in top k chunk then we take the reciprocal of it (i.e. top 3 chunk then 1/3 = 0.33). 
                            # Lastly we sum and take the mean of the overall reciprocal rank (i.e. Query 1: top 1 = 1/1 = 1; Query 2: top 3 = 1/3 = 0.33; Query 3: top 5 = 1/5 = 0.2; Mean Reciprocal Rank = 1+0.33+0.2/3 = 0.51)

    for sample in dataset: # Loop through all sample question from json file
        results = retriever.retrieve(sample["question"], top_k=max(K_VALUES)) # Retrieve top 5 relevant chunks based on the sample question
        expected_filename = sample["expected_filename"] # Relevant filename for user query
        expected_pages = sample["expected_pages"] # Relevant page of filename for user query
        relevant_ranks = []

        print()
        print("=" * 60)
        print(f"QUERY: {sample['question']}")
        print(f"EXPECTED FILE: {expected_filename}")
        print(f"EXPECTED PAGES: {expected_pages}")

        for rank, result in enumerate(results, start=1): # Start loop at 1
            if is_relevant(result, expected_filename, expected_pages): # If the retrieved chunk from vector DB is relevant to the user query, we append the chunk into ranks. We determine if the chunk is relevant by providing the expected page and document that the vector DB should retrieve
                relevant_ranks.append(rank)

        for k in K_VALUES: # Loop through k = 1, k = 3, k = 5
            top_k_results = results[:k] # Get the top_1, top_3, top_5 results by slicing result list (i.e. k = 1, results[:1])
            relevant_results = [result for result in top_k_results if is_relevant(result, expected_filename, expected_pages)] # If top results pass relevant check, then only added into relevant_result list

            # i.e. k = 3, top_3_result = results[:3] which only give 3 result since running index starts at 1
            # Check if the top_3_result chunk passes relevant check (if the chunks filename & pages are the same as the expected ones)
            # If lets say 2 of the 3 chunks are relevant which has the same filename & page as expected retrieved answer, and the expected relevant pages only has 2
            # hit@3 += 1, precision@3 += 2/3, recall@3 += 2/2

            # Hit@K
            if relevant_results:
                hit_sums[k] += 1

            # Precision@K
            precision_sums[k] += (len(relevant_results) / k)

            # Page-Level Recall@K
            retrieved_relevant_pages = {
                result.page_number for result in top_k_results if result.filename == expected_filename and result.page_number in expected_pages
            }

            if expected_pages:
                page_recall_sums[k] += (len(retrieved_relevant_pages) / len(expected_pages))

        # Reciprocal Rank
        if relevant_ranks:
            first_relevant_rank = min(relevant_ranks)
            reciprocal_rank_sum += (1 / first_relevant_rank)

    # Average over all queries
    total = len(dataset)
    for k in K_VALUES:
        print(f"Hit@{k}: {hit_sums[k] / total:.3f}")

        print(
            f"Precision@{k}: {precision_sums[k] / total:.3f}")

        print(
            f"PageRecall@{k}: {page_recall_sums[k] / total:.3f}")

    print(
        f"MRR: {reciprocal_rank_sum / total:.3f}")