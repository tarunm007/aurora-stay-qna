
import streamlit as st
import requests
import os


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# TM: Use a container/service URL supplied by the deployment environment while
# TM: retaining localhost for local development. Avoid doubled path separators.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7860").strip().rstrip("/")
# TM: Bound connection setup and response waiting below Gunicorn's 120s timeout.
REQUEST_TIMEOUT = (5, 110)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Aurora Stays",
    page_icon="🏨",
    layout="wide"
)


# ---------------------------------------------------------
# Sidebar  -  Model parameters
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Parameters")

    k = st.slider(
        "k (chunks to retrieve)",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of relevant chunks fetched from the vector store."
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher values make the output more creative; lower values make it more focused."
    )

    max_tokens = st.slider(
        "Max tokens",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
        help="Maximum number of tokens in the generated answer."
    )


# ---------------------------------------------------------
# Main area
# ---------------------------------------------------------

st.title("🏨 Aurora Stays")
st.write("Ask a question about Aurora Stays policies.")


# ---------------------------------------------------------
# Response type selection (two options only)
# ---------------------------------------------------------

response_type = st.radio(
    "Response mode",
    ["Answer only", "Answer + Relevant Chunks"],
    horizontal=True
)


# ---------------------------------------------------------
# Query input
# ---------------------------------------------------------

query = st.text_input(
    "Your question",
    placeholder="e.g., What is the cancellation policy for a flexible booking?"
)


# ---------------------------------------------------------
# Submit
# ---------------------------------------------------------

if st.button("Submit", type="primary"):

    if not query.strip():
        st.warning("Please enter a question.")

    else:

        # Pick endpoint
        if response_type == "Answer only":
            endpoint = "/v1/answer"
        else:
            endpoint = "/v1/answer_with_relevant_chunks"

        # Build request payload with parameters
        payload = {
            "query": query,
            "k": k,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            with st.spinner("Thinking…"):
                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )

            if response.status_code == 200:

                result = response.json()
                # TM: Reject malformed responses instead of crashing on missing
                # TM: answer/chunk fields when a proxy or backend returns bad data.
                if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
                    raise ValueError("Invalid answer response")
                if response_type == "Answer + Relevant Chunks" and (
                    not isinstance(result.get("relevant_chunks"), list)
                    or not all(isinstance(chunk, str) for chunk in result["relevant_chunks"])
                ):
                    raise ValueError("Invalid relevant-chunks response")

                # -- Answer only --------------------------------
                if response_type == "Answer only":

                    st.subheader("Answer")
                    st.write(result["answer"])

                # -- Answer + Relevant Chunks -------------------
                else:

                    st.subheader("Answer")
                    st.write(result["answer"])

                    st.subheader("Relevant Chunks")
                    for i, chunk in enumerate(result["relevant_chunks"], start=1):
                        with st.expander(f"Chunk {i}"):
                            st.write(chunk)

            else:
                # TM: Show the backend's safe validation message when available;
                # TM: do not display raw HTML, stack traces, or proxy error pages.
                error_message = None
                try:
                    error_body = response.json()
                    if isinstance(error_body, dict) and isinstance(error_body.get("error"), str):
                        error_message = error_body["error"]
                except ValueError:
                    pass
                st.error(
                    f"Unable to get a response from the RAG API (HTTP {response.status_code}). "
                    + (error_message or "Please try again later.")
                )

        # TM: Distinguish slow responses from connection failures and malformed
        # TM: JSON so the user sees the relevant failure without another API call.
        except requests.exceptions.Timeout:
            st.error("The RAG API took too long to respond. Please try again.")
        except requests.exceptions.JSONDecodeError:
            st.error("The RAG API returned an invalid response. Please try again later.")
        except requests.exceptions.RequestException:
            st.error(
                "Unable to connect to the RAG API. "
                "Please make sure the backend service is running."
            )
        except ValueError:
            st.error("The RAG API returned an invalid response. Please try again later.")
