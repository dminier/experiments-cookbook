class EN:
    DEFAULT_QUERY_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    """

    DEFAULT_ANSWER_PROMPT = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """

    DEFAULT_DOCUMENT_PROMPT = "{page_content}"


class FR:
    DEFAULT_QUERY_PROMPT = """Compte tenu de la conversation suivante et d'une question de suivi, reformulez la question de suivi pour en faire une question autonome, dans la langue française.
    Historique des discussions :
    {chat_history}
    Entrée de suivi : {question}
    Question autonome :
    """

    DEFAULT_ANSWER_PROMPT = """Répondez à la question en vous basant uniquement sur le contexte suivant. La réponse doit être rédigée en français sans le mentionner.:
    {context}
    Question : {question}
    """

    DEFAULT_DOCUMENT_PROMPT = "{page_content}"
