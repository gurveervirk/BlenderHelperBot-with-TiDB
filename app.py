import streamlit as st
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.core.prompts import ChatMessage
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings, VectorStoreIndex

conn_string = st.secrets["database"]["connection_string"]
ca_cert = st.secrets["certificates"]["ca_cert"]

with open("ca_cert.pem", "w") as cert_file:
    cert_file.write(ca_cert)

def init(hf_api_key):
    Settings.embed_model = HuggingFaceInferenceAPIEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1", token=hf_api_key)
    Settings.llm = HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-Instruct-v0.3", token=hf_api_key, task="TGI", num_output=1024, context_window=4096)
    llm = Settings.llm

    VECTOR_TABLE_NAME = "default"
    tidbvec = TiDBVectorStore(
        connection_string=conn_string + "?ssl_ca=ca_cert.pem",
        table_name=VECTOR_TABLE_NAME,
        distance_strategy="cosine",
        vector_dimension=1024,
        drop_existing_table=False
    )
    vector_index = VectorStoreIndex.from_vector_store(vector_store=tidbvec)
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

    chat_engine = vector_index.as_chat_engine(chat_mode="condense_plus_context", llm=llm,
        context_prompt=(
            "You are a chatbot, who needs to answer questions, preferably using the provided context"
            "Here are the relevant documents for the context:\n"
            "{context_str}"
            "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
        ), memory=memory, verbose=True, similarity_top_k=7
    )

    return chat_engine, memory

def chat(chat_engine, memory, query):
    memory.put(ChatMessage.from_str(content=query))
    response = chat_engine.chat(query).response
    return response

# Initialize global variables
def main():
    chat_engine = None
    memory = None
    with st.sidebar:
        hf_api_key = st.text_input("HF API Key", key="chatbot_api_key", type="password")

        if hf_api_key:
            chat_engine, memory = init(hf_api_key)
            st.success("HF API key added successfully.")

    st.title("💬 Blender Helper Bot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not hf_api_key:
            st.info("Please add your HF API key to continue.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = chat(chat_engine, memory, prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == '__main__':
    main()
