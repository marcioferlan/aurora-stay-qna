
import streamlit as st
import requests


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Both containers run with --network host inside the Codespace, so the
# frontend reaches the Flask backend on the Codespace's own localhost.
BACKEND_URL = "http://localhost:7860"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AuroraStay QnA",
    page_icon="\U0001F3E8",
    layout="wide"
)


# ---------------------------------------------------------
# Sidebar  -  Model parameters
# ---------------------------------------------------------

with st.sidebar:

    st.header("\u2699\uFE0F Parameters")

    k = st.slider(
        "k (chunks to retrieve)",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of policy chunks fetched from the vector store. Four is the value the pipeline was evaluated with."
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Keep this at 0 so the same policy question always gets the same answer."
    )

    max_tokens = st.slider(
        "Max tokens",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
        help="Maximum length of the generated answer."
    )

    st.divider()
    st.caption(
        "Internal tool for AuroraStay front-desk and reservations staff. "
        "Answers are generated only from the four AuroraStay policy documents. "
        "Always check the supporting sources before quoting a policy to a guest."
    )


# ---------------------------------------------------------
# Main area
# ---------------------------------------------------------

st.title("\U0001F3E8 AuroraStay QnA")
st.write("Ask a question about AuroraStay booking terms, room and property information, refunds and charges, or guest conduct policies.")


# ---------------------------------------------------------
# Response type selection (two options only)
# ---------------------------------------------------------

response_type = st.radio(
    "Response mode",
    ["Answer + Relevant Chunks", "Answer only"],
    horizontal=True,
    help="Keep the sources visible while the system is being validated."
)


# ---------------------------------------------------------
# Query input
# ---------------------------------------------------------

query = st.text_input(
    "Your question",
    placeholder="e.g., How late can a guest cancel a Flexible rate without being charged?"
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
            with st.spinner("Searching the policy documents\u2026"):
                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json=payload
                )

            if response.status_code == 200:

                result = response.json()

                # -- Answer only --------------------------------
                if response_type == "Answer only":

                    st.subheader("Answer")
                    st.write(result["answer"])

                # -- Answer + Relevant Chunks -------------------
                else:

                    st.subheader("Answer")
                    st.write(result["answer"])

                    st.subheader("Supporting sources")
                    st.caption(
                        f"{len(result['relevant_chunks'])} policy passages were retrieved and used to produce the answer above."
                    )
                    for i, chunk in enumerate(result["relevant_chunks"], start=1):
                        with st.expander(f"Source passage {i}"):
                            st.write(chunk)

            else:
                st.error(
                    f"Unable to get a response from the RAG API. "
                    f"Status code: {response.status_code}"
                )

        except requests.exceptions.RequestException:
            st.error(
                "Unable to connect to the RAG API. "
                "Please make sure the backend service is running."
            )
