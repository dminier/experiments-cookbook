def ollama_config(model_name="mistral"):
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms.ollama import Ollama

    ollama_llm = Ollama(model=model_name)
    ollama_llm_embeddings = OllamaEmbeddings(model=model_name)
    ollama_chat = ChatOllama(model=model_name)
    ollama_chat_strict = ChatOllama(model=model_name, temperature=0)

    return ollama_llm, ollama_llm_embeddings, ollama_chat, ollama_chat_strict
