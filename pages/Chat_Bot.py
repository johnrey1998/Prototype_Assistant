import streamlit as st
from RAGChain import ask_chain, wipe_session_history

pcieerd_logo = "pcieerd_logo.png"
user_id = str(123)
conversation_id = str(123)
sources = ""
st.title("💬 Chatbot")
st.caption("🚀 Source: DOST-PCIEERD Database")

st.sidebar.image(pcieerd_logo, caption="DOST-PCIEERD")

if "reset" not in st.session_state:
    st.session_state["reset"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]



for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])




if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = ask_chain(prompt, user_id, conversation_id)
    answer = response['answer']
    sources = response["context"]
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)



def click_button():
    del(st.session_state.messages)
    st.session_state.reset = True



if st.session_state["reset"]:
    wipe_session_history(user_id, conversation_id)
    st.session_state["reset"] = False


with st.container():
    with st.expander("Sources: "):
        for source in sources:
            st.write(source)
    with st.expander("Raw Sources: "):
        for source in sources:
            st.write_stream(source)

    st.button("Reset memory", on_click=click_button)
    st.caption("user_id: " + user_id + ", conversation_id: " + conversation_id)



