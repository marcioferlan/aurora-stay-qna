
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context.
# The rules below are identical to the ones used in the notebook, so the deployed
# service behaves exactly like the pipeline that was evaluated.
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly present in the retrieved policy information.
    2. Do not use outside knowledge and do not infer, assume, or generalize beyond what is stated.
    3. Include every condition that applies: deadlines, cut-off times, fee amounts, taxes, occupancy limits, and approval requirements.
    4. Do not omit exceptions, exclusions, limitations, or property-specific rules that change the answer.
    5. When a rule applies to a specific property, name that property in the answer.
    6. Leave out retrieved information that is not relevant to the question that was asked.
    7. Answer the question directly in the first sentence.
    8. Keep the answer concise and in plain language a front-desk agent can read aloud to a guest.
    9. End the answer with "Source: " followed by the document name taken from the chunk metadata.
    10. If the retrieved information does not cover the question, reply exactly:
        "The available AuroraStay policy documents do not contain enough information to answer this question."
    11. If the retrieved information conflicts, state the conflict explicitly instead of choosing one side.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
