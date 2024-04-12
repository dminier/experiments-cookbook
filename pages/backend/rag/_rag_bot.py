from operator import itemgetter

from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from pages.backend.infrastructure import (LLM_CHAT_STRICT,
                                          LLM_CHAT,
                                          WEB_LOADER)
from pages.backend.rag.default_prompt import EN


class RetrievalAugmentedGenerationChatBot:

    def __init__(self, query_prompt_template, answer_prompt_template,
                 document_prompt_template):
        self.memory = ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )
        self.memory.memory_key = "chat_history"
        self.chain = self._build_rag(query_prompt_template, answer_prompt_template,
                                     document_prompt_template)

    def _build_rag(self,
                   query_prompt_template=EN.DEFAULT_QUERY_PROMPT,
                   answer_prompt_template=EN.DEFAULT_ANSWER_PROMPT,
                   document_prompt_template=EN.DEFAULT_DOCUMENT_PROMPT,
                   ):
        condense_question_prompt = PromptTemplate.from_template(query_prompt_template)

        answer_prompt = ChatPromptTemplate.from_template(answer_prompt_template)
        doc_prompt = PromptTemplate.from_template(template=document_prompt_template)

        def _combine_documents(
                docs, document_prompt=doc_prompt, document_separator="\n\n"
        ):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)

        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("chat_history"),
        )

        standalone_question = {
            "standalone_question": {
                                       "question": lambda x: x["question"],
                                       "chat_history": lambda x: get_buffer_string(x["chat_history"]),
                                   } | condense_question_prompt | LLM_CHAT_STRICT | StrOutputParser(),
        }

        retriever = WEB_LOADER.as_retriever()
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
            "answer": final_inputs | answer_prompt | LLM_CHAT,
            "docs": itemgetter("docs"),
        }

        return loaded_memory | standalone_question | retrieved_documents | answer
