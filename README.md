# AuroraStay QnA

Retrieval-Augmented Generation service that answers AuroraStay hotel policy
questions for internal staff, grounded only in the four AuroraStay policy
documents.

## Layout

    backend/
      app.py             Flask API (gunicorn, port 7860)
      rag_utils.py       retrieve / generate / rag
      faiss_index/       serialized FAISS index (pre-computed embeddings)
      requirements.txt
      Dockerfile
    frontend/
      app.py             Streamlit UI (port 8501)
      requirements.txt
      Dockerfile

## Endpoints

    GET  /                             health check
    POST /v1/relevant_chunks           retrieved chunks only
    POST /v1/answer                    generated answer only
    POST /v1/answer_with_relevant_chunks   answer plus its source chunks

## Running in GitHub Codespaces

Add `OPENAI_API_KEY` and `OPENAI_API_BASE` as Codespaces secrets, then:

    cd backend
    docker build -t aurora-backend .
    docker run --name backend --network host \
      -e OPENAI_API_KEY="$OPENAI_API_KEY" \
      -e OPENAI_API_BASE="$OPENAI_API_BASE" aurora-backend

    cd ../frontend
    docker build -t aurora-frontend .
    docker run --name frontend --network host aurora-frontend

Both containers use the host network, so the frontend reaches the backend on
localhost:7860. Set ports 7860 and 8501 to Public in the Ports tab.
