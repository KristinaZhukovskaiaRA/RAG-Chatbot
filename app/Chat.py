import streamlit as st
from services.ai_service import AIService

st.set_page_config(page_title="Chat", page_icon="💬")

st.title("💬 Chat with Your Documents")

if "ai_service" not in st.session_state:
    st.session_state.ai_service = AIService()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if st.session_state.ai_service.retriever is None:
            full_response = "I'm sorry, but I don't have any documents to reference. Please upload some documents or URLs first."
            message_placeholder.markdown(full_response)

        else:
            for chunk in st.session_state.ai_service.chain.stream(
                {
                    "question": prompt,
                    "context": "\n\n".join(
                        [
                            doc.page_content
                            for doc in st.session_state.ai_service.retriever.get_relevant_documents(
                                prompt
                            )
                        ]
                    ),
                    "chat_history": st.session_state.ai_service.chat_history,
                }
            ):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.session_state.ai_service.chat_history.append({"role": "user", "content": prompt})
    st.session_state.ai_service.chat_history.append(
        {"role": "assistant", "content": full_response}
    )
