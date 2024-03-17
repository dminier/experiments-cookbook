from operator import itemgetter

from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from pages.langchain.document.vector_retriever import retriever
from pages.langchain.llm import chat_strict, chat

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)
query_prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(query_prompt_template)

answer_prompt_template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_prompt_template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)

# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
                               "question": lambda x: x["question"],
                               "chat_history": lambda x: get_buffer_string(x["chat_history"]),
                           }
                           | CONDENSE_QUESTION_PROMPT
                           | chat_strict
                           | StrOutputParser(),
}

# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | chat,
    "docs": itemgetter("docs"),
}
# And now we put it all together!
RAG_CHAIN = loaded_memory | standalone_question | retrieved_documents | answer
