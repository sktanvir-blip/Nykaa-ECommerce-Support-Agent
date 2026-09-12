from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
import chromadb

#calibrated similarity threshold
SIMILARITY_THRESHOLD = 0.30

KNOWLEDGE_BASE_DIR = Path("Knowledge_base") 


def load_documents():
    documents = []

    for file_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")): 

        text = file_path.read_text(encoding="utf-8").strip()

        document_id = file_path.stem.split("_")[0]
        topic = "_".join(file_path.stem.split("_")[1:])

        document = {
            "document_id": document_id,
            "source": file_path.name,
            "topic": topic,
            "text": text,
        }

        documents.append(document)

    return documents
    
def fixed_size_chunking(
    documents,
    chunk_size=300,
    overlap=50
):
    chunks = []

    for document in documents:

        text = document["text"]

        start = 0
        chunk_number = 1

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            chunk = {
                "chunk_id": (
                    f'{document["document_id"]}'
                    f'_FIXED_{chunk_number}'
                ),
                "document_id": document["document_id"],
                "source": document["source"],
                "topic": document["topic"],
                "chunking_strategy": "fixed",
                "text": chunk_text,
            }

            chunks.append(chunk)

            chunk_number += 1

            if end >= len(text):
                break

            start = end - overlap

    return chunks


def sentence_based_chunking(documents):
    chunks = []

    for document in documents:

        sentences = re.split(
            r'(?<=[.!?])\s+',
            document["text"]
        )

        for index, sentence in enumerate(sentences, start=1):

            sentence = sentence.strip()

            if not sentence:
                continue

            chunk = {
                "chunk_id": (
                    f'{document["document_id"]}'
                    f'_SENTENCE_{index}'
                ),
                "document_id": document["document_id"],
                "source": document["source"],
                "topic": document["topic"],
                "chunking_strategy": "sentence",
                "text": sentence,
            }

            chunks.append(chunk)

    return chunks

def create_embeddings(chunks, model):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings

def create_chroma_collections():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    fixed_collection = client.get_or_create_collection(
        name="nykaa_fixed_chunks",
        metadata={"hnsw:space": "cosine"}
    )

    sentence_collection = client.get_or_create_collection(
        name="nykaa_sentence_chunks",
        metadata={"hnsw:space": "cosine"}
    )

    return fixed_collection, sentence_collection

def index_chunks(collection, chunks, embeddings):

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "document_id": chunk["document_id"],
            "source": chunk["source"],
            "topic": chunk["topic"],
            "chunking_strategy": chunk["chunking_strategy"], 
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

def search_collection(
    collection,
    query,
    model,
    top_k=3
):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    return results

def grounded_retrieval(
    collection,
    query,
    model,
    threshold=SIMILARITY_THRESHOLD,
    top_k=3
):
    results = search_collection(
        collection,
        query,
        model,
        top_k=top_k
    )

    top_distance = results["distances"][0][0]

    top_similarity = calculate_similarity(
        top_distance
    )

    if top_similarity < threshold:
        return {
            "grounded": False,
            "similarity": top_similarity,
            "context": [],
            "results": results,
        }

    context = []

    for i in range(len(results["documents"][0])):

        context.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "document_id": results["metadatas"][0][i]["document_id"],
            "topic": results["metadatas"][0][i]["topic"],
            "similarity": calculate_similarity(
                results["distances"][0][i]
            ),
        })

    return {
        "grounded": True,
        "similarity": top_similarity,
        "context": context,
        "results": results,
    }

def generate_grounded_answer(
    query,
    retrieval_result
):

    if not retrieval_result["grounded"]:
        return {
            "answer": (
                "I don't know based on the available "
                "knowledge base."
            ),
            "grounded": False,
            "sources": [],
        }

    context = retrieval_result["context"]

    context_text = "\n".join(
        item["text"]
        for item in context
    )

    answer = (
        "Based only on the retrieved knowledge base:\n\n"
        f"{context_text}"
    )

    sources = list(
        dict.fromkeys(
            item["source"]
            for item in context
        )
    )

    return {
        "answer": answer,
        "grounded": True,
        "sources": sources,
    }

def calculate_similarity(distance):
    return 1 - distance

def calibrate_similarity(
    collection,
    model,
    queries,
    collection_name
):

    print(
        f"\n{collection_name} CALIBRATION"
    )

    scores = []

    for query in queries:

        results = search_collection(
            collection,
            query,
            model,
            top_k=1
        )

        distance = results["distances"][0][0]

        similarity = calculate_similarity(
            distance
        )

        scores.append(similarity)

        print(
            f"\nQuery: {query}"
        )

        print(
            f"Distance: {distance:.4f}"
        )

        print(
            f"Similarity: {similarity:.4f}"
        )

        print(
            "Source:",
            results["metadatas"][0][0]["source"]
        )

    return scores

def print_search_results(
    collection_name,
    results
):

    print(
        f"\n{collection_name}"
    )

    for i in range(len(results["ids"][0])):

        print(
            f"\nResult {i + 1}"
        )

        print(
            "ID:",
            results["ids"][0][i]
        )

        print(
            "Distance:",
            results["distances"][0][i]
        )

        print(
            "Metadata:",
            results["metadatas"][0][i]
        )

        print(
            "Text:",
            results["documents"][0][i]
        )

def run_grounded_tests(
    collection,
    model,
    queries,
    threshold=SIMILARITY_THRESHOLD
):

    print("\nGROUNDED GENERATION TESTS")

    for number, query in enumerate(queries, start=1):

        result = grounded_retrieval(
            collection,
            query,
            model,
            threshold=threshold,
            top_k=3
        )

        answer = generate_grounded_answer(
            query,
            result
        )

        print(f"\n--- Test {number} ---")

        print("Query:", query)

        print(
            f"Top similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            "Grounded:",
            answer["grounded"]
        )

        print(
            "Sources:",
            answer["sources"]
        )

        print(
            "Answer:",
            answer["answer"]
        )

def evaluate_chunking_strategy(
    collection,
    model,
    queries,
    expected_documents,
    collection_name,
    top_k=3
):

    print(
        f"\n{collection_name} EVALUATION"
    )

    total_precision = 0
    total_recall = 0

    for number, query in enumerate(queries, start=1):

        results = search_collection(
            collection,
            query,
            model,
            top_k=top_k
        )

        retrieved_document_ids = list(
            dict.fromkeys(
                metadata["document_id"]
                for metadata in results["metadatas"][0]
            )
        )

        expected_document = expected_documents[number - 1]

        relevant_retrieved = (
            1
            if expected_document in retrieved_document_ids
            else 0
        )

        total_retrieved = len(
            retrieved_document_ids
        )

        total_relevant = 1

        precision = (
            relevant_retrieved / total_retrieved
            if total_retrieved > 0
            else 0
        )

        recall = (
            relevant_retrieved / total_relevant
        )

        total_precision += precision
        total_recall += recall

        print(f"\n--- Query {number} ---")
        print("Query:", query)

        print(
            "Expected document:",
            expected_document
        )

        print(
            "Retrieved documents:",
            retrieved_document_ids
        )

        print(
            f"Precision: "
            f"{relevant_retrieved}/{total_retrieved} "
            f"= {precision:.4f}"
        )

        print(
            f"Recall: "
            f"{relevant_retrieved}/{total_relevant} "
            f"= {recall:.4f}"
        )

    average_precision = (
        total_precision / len(queries)
    )

    average_recall = (
        total_recall / len(queries)
    )

    print(
        f"\nAverage Precision: "
        f"{average_precision:.4f}"
    )

    print(
        f"Average Recall: "
        f"{average_recall:.4f}"
    )

    return {
        "precision": average_precision,
        "recall": average_recall,
    }

if __name__ == "__main__":

    documents = load_documents()

    fixed_chunks = fixed_size_chunking(documents)

    sentence_chunks = sentence_based_chunking(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Fixed-size chunks: {len(fixed_chunks)}")
    print(f"Sentence-based chunks: {len(sentence_chunks)}")

    print("\nDocument lengths:")

    for document in documents:
        print(
            document["document_id"],
            "|",
            document["source"],
            "|",
            len(document["text"]),
            "characters"
        )

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")

    fixed_embeddings = create_embeddings(
        fixed_chunks,
        embedding_model
    )

    sentence_embeddings = create_embeddings(
        sentence_chunks,
        embedding_model
    )

    print(
        "\nFixed embeddings shape:",
        fixed_embeddings.shape
    )

    print(
        "Sentence embeddings shape:",
        sentence_embeddings.shape
    )

    fixed_collection, sentence_collection = (
        create_chroma_collections()
    )

    index_chunks(
        fixed_collection,
        fixed_chunks,
        fixed_embeddings
    )

    index_chunks(
        sentence_collection,
        sentence_chunks,
        sentence_embeddings
    )

    print("\nChromaDB indexing completed.")

    print(
        "Fixed collection count:",
        fixed_collection.count()
    )

    print(
        "Sentence collection count:",
        sentence_collection.count()
    )

    in_scope_queries = [
        "How many days can I return footwear?",
        "How long does a COD refund take?",
        "What is the delivery SLA?"
    ]

    out_of_scope_queries = [
        "What is the weather in Mumbai today?",
        "Who won the last cricket World Cup?"
    ]

    print(
        "\n\nTHRESHOLD CALIBRATION"
    )

    fixed_in_scope_scores = calibrate_similarity(
        fixed_collection,
        embedding_model,
        in_scope_queries,
        "FIXED-SIZE"
    )

    fixed_out_of_scope_scores = calibrate_similarity(
        fixed_collection,
        embedding_model,
        out_of_scope_queries,
        "FIXED-SIZE"
    )

    sentence_in_scope_scores = calibrate_similarity(
        sentence_collection,
        embedding_model,
        in_scope_queries,
        "SENTENCE-BASED"
    )

    sentence_out_of_scope_scores = calibrate_similarity(
        sentence_collection,
        embedding_model,
        out_of_scope_queries,
        "SENTENCE-BASED"
    )
    test_query = (
        "What is the weather in mumbai today?"
    )

    print(
        f"\nTest query: {test_query}"
    )

    fixed_results = search_collection(
        fixed_collection,
        test_query,
        embedding_model,
        top_k=3
    )

    sentence_results = search_collection(
        sentence_collection,
        test_query,
        embedding_model,
        top_k=3
    )

    print_search_results(
        "FIXED-SIZE COLLECTION",
        fixed_results
    )

    print_search_results(
        "SENTENCE-BASED COLLECTION",
        sentence_results
    )

    grounded_result = grounded_retrieval(
        sentence_collection,
        test_query,
        embedding_model,
        threshold=SIMILARITY_THRESHOLD,
        top_k=3
    )

    final_answer = generate_grounded_answer(
        test_query,
        grounded_result
    )

    print("\nGROUNDED ANSWER")

    print(
        "Grounded:",
        final_answer["grounded"]
    )

    print(
        "Top similarity:",
        grounded_result["similarity"]
    )

    print(
        "Answer:",
        final_answer["answer"]
    )

    print(
        "Sources:",
        final_answer["sources"]
    )


    grounded_test_queries = [
        "How many days can I return footwear?",
        "How long does a COD refund take?",
        "What is the delivery SLA?",
        "Can I get a size exchange for footwear?",
        "What is the process for claiming a damaged item?"
    ]

    run_grounded_tests(
        sentence_collection,
        embedding_model,
        grounded_test_queries
    )

    out_of_scope_query = (
        "What is the weather in Mumbai today?"
    )

    out_of_scope_result = grounded_retrieval(
        sentence_collection,
        out_of_scope_query,
        embedding_model,
        threshold=SIMILARITY_THRESHOLD,
        top_k=3
    )

    out_of_scope_answer = generate_grounded_answer(
        out_of_scope_query,
        out_of_scope_result
    )

    print("\nOUT-OF-SCOPE TEST")

    print(
        "Query:",
        out_of_scope_query
    )

    print(
        f"Top similarity: "
        f"{out_of_scope_result['similarity']:.4f}"
    )

    print(
        "Grounded:",
        out_of_scope_answer["grounded"]
    )

    print(
        "Answer:",
        out_of_scope_answer["answer"]
    )

    print(
        "Sources:",
        out_of_scope_answer["sources"]
    )

    evaluation_queries = [
        "How many days can I return footwear?",
        "How long does a COD refund take?",
        "What is the delivery SLA?",
        "Can I get a size exchange for footwear?",
        "What is the process for claiming a damaged item?"
    ]

    expected_documents = [
        "01",
        "02",
        "03",
        "09",
        "10"
    ]

    fixed_evaluation = evaluate_chunking_strategy(
        fixed_collection,
        embedding_model,
        evaluation_queries,
        expected_documents,
        "FIXED-SIZE",
        top_k=3
    )

    sentence_evaluation = evaluate_chunking_strategy(
        sentence_collection,
        embedding_model,
        evaluation_queries,
        expected_documents,
        "SENTENCE-BASED",
        top_k=3
    )


    print(
        "Fixed-size average precision:",
        f"{fixed_evaluation['precision']:.4f}"
    )

    print(
        "Fixed-size average recall:",
        f"{fixed_evaluation['recall']:.4f}"
    )

    print(
        "Sentence-based average precision:",
        f"{sentence_evaluation['precision']:.4f}"
    )

    print(
        "Sentence-based average recall:",
        f"{sentence_evaluation['recall']:.4f}"
    )