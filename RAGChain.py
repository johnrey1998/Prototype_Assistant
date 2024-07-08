from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableWithMessageHistory, \
    ConfigurableFieldSpec
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv
load_dotenv()


CHROMA_PATH = "chroma"
embd = OpenAIEmbeddings()
model = ChatOpenAI(model="gpt-4o", temperature=0)
qa_system_prompt = """You are an assistant for question-answering tasks.
EXCLUSIVELY use ONLY the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
contextualize_q_system_prompt = """Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

store = {}

def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id,conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


def ask_chain(query, user_id, conversation_id):
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embd)
    retriever = vectorstore.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="Unique identifier for the user.",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="conversation_id",
                annotation=str,
                name="Conversation ID",
                description="Unique identifier for the conversation.",
                default="",
                is_shared=True,
            ),
        ]
    )
    result = conversational_rag_chain.invoke(
        {"input": query},
        {"configurable": {"user_id": user_id, "conversation_id": conversation_id}},
    )
    return result

def wipe_session_history(user_id: str, conversation_id: str):
    try:
        del store[(user_id, conversation_id)]
        print("Session history wiped successfully.")
    except KeyError:
        print("Key not found in store.")



if __name__ == "__main__":
    query = "What is the most important statement?"
    user_id = "admin"
    conversation_id = "root"
    result = ask_chain(query, user_id, conversation_id)
    print(result)
    print(store)