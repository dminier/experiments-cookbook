docker compose up -d
docker compose exec ollama ollama pull mistral

streamlit run app.py