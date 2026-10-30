
# Import necessary libraries
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from pathlib import Path

# Import RAG utility functions
from rag_utils import DEFAULT_K, retrieve, rag


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
# TM: Resolve the packaged index relative to this module, not the process's
# TM: working directory, so both Gunicorn and local validation load the same data.
VECTORSTORE_PATH = str(Path(__file__).resolve().parent / "faiss_index")


# ---------------------------------------------------------
# Initialize the Flask application
# ---------------------------------------------------------

rag_api = Flask("Aurora Stay API")


# ---------------------------------------------------------
# Load the embedding model
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ---------------------------------------------------------
# Load the saved FAISS vector store
# ---------------------------------------------------------

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Vectors stored: {vectorstore.index.ntotal}")


# TM: Reject malformed inputs before retrieval or paid model calls. All three
# TM: POST endpoints share the evaluated defaults and this validation boundary.
@rag_api.before_request
def validate_api_request():
    if request.method != "POST" or request.path not in {
        "/v1/relevant_chunks", "/v1/answer_with_relevant_chunks", "/v1/answer"
    }:
        return None
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required."}), 400
    if not isinstance(data.get("query"), str) or not data["query"].strip():
        return jsonify({"error": "query must be a non-empty string."}), 400
    requested_k = data.get("k", DEFAULT_K)
    if type(requested_k) is not int or not 1 <= requested_k <= 20:
        return jsonify({"error": "k must be an integer between 1 and 20."}), 400
    for parameter, default, lower, upper in (
        ("temperature", 0.0, 0.0, 2.0), ("top_p", 1.0, 0.0, 1.0)
    ):
        value = data.get(parameter, default)
        if type(value) not in (int, float) or not lower <= value <= upper:
            return jsonify({"error": f"{parameter} must be a number between {lower} and {upper}."}), 400
    token_limit = data.get("max_tokens", 512)
    if type(token_limit) is not int or not 1 <= token_limit <= 4096:
        return jsonify({"error": "max_tokens must be an integer between 1 and 4096."}), 400
    if not isinstance(data.get("model_name", "gpt-4o-mini"), str) or not data.get("model_name", "gpt-4o-mini").strip():
        return jsonify({"error": "model_name must be a non-empty string."}), 400
    return None


# TM: A local health probe checks loaded artifacts only. It never contacts an
# TM: LLM or exposes API credentials, and can be used by Docker HEALTHCHECK.
@rag_api.get("/health")
def health():
    vector_count = vectorstore.index.ntotal
    return jsonify({"status": "ok" if vector_count else "not_ready", "vectors": vector_count}), 200 if vector_count else 503


# ---------------------------------------------------------
# Default route
# ---------------------------------------------------------

@rag_api.get("/")
def home():
    """
    Handles GET requests to the root URL.
    """
    return "Welcome to the Aurora Stay RAG API!"


# ---------------------------------------------------------
# Relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    """
    Handles POST requests to /v1/relevant_chunks.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 3
    }

    Returns the top-k relevant chunks retrieved from
    the vector store.
    """

    # Get JSON data from request body
    data = request.get_json()

    # Extract parameters
    query = data["query"]
    # TM: Match the notebook's evaluated K=4 instead of the old API default of 2.
    k = data.get("k", DEFAULT_K)

    # Retrieve relevant chunks
    # TM: Enable the same search planning, hybrid retrieval, source validation,
    # TM: and bounded repair used by rag(). This retrieval-only endpoint can now
    # TM: make model calls; only its response omits the final generated answer.
    retrieval_model = ChatOpenAI(
        model=data.get("model_name", "gpt-4o-mini"),
        temperature=0,
        top_p=1,
        max_tokens=512,
    )
    chunks = retrieve(
        query=query,
        k=k,
        vectorstore=vectorstore,
        model=retrieval_model
    )

    # Return the retrieved chunks
    return jsonify({
        "query": query,
        "k": k,
        "relevant_chunks": chunks
    })


# ---------------------------------------------------------
# Answer with relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    """
    Handles POST requests to /v1/answer_with_relevant_chunks.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 3,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512
    }

    Returns both the generated answer and the relevant
    chunks used to generate the answer.
    """

    # Get JSON data from request body
    data = request.get_json()

    # Extract parameters
    query = data["query"]
    # TM: Use the same retrieval default as the evaluated notebook pipeline.
    k = data.get("k", DEFAULT_K)
    model_name = data.get("model_name", "gpt-4o-mini")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 512)

    # Run the complete RAG pipeline
    answer, chunks = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore
    )

    # Return answer, chunks, and parameters used
    return jsonify({
        "query": query,
        "k": k,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "relevant_chunks": chunks,
        "answer": answer
    })


# ---------------------------------------------------------
# Answer endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer")
def answer():
    """
    Handles POST requests to /v1/answer.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 3,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512
    }

    Returns only the generated answer.
    """

    # Get JSON data from request body
    data = request.get_json()

    # Extract parameters
    query = data["query"]
    # TM: Use the same retrieval default as the evaluated notebook pipeline.
    k = data.get("k", DEFAULT_K)
    model_name = data.get("model_name", "gpt-4o-mini")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 512)

    # Run the complete RAG pipeline
    generated_answer, _ = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore
    )

    # Return only the answer
    return jsonify({
        "answer": generated_answer
    })


# ---------------------------------------------------------
# Run the Flask application
# ---------------------------------------------------------

if __name__ == "__main__":
    # TM: Direct execution is a local fallback; Docker starts Gunicorn instead.
    rag_api.run(host="0.0.0.0", port=7860, debug=False)
