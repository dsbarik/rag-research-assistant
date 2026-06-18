import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.getcwd())
from src import ResearchAssistant

# --- Page Config ---
st.set_page_config(page_title="RAG Research Assistant", page_icon="🤖", layout="wide")

# --- Initialize App Backend ---
# Using @st.cache_resource so the database and models don't reload on every button click
@st.cache_resource
def get_app():
    return ResearchAssistant()

app = get_app()

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.conversation_id = None

# --- Sidebar: Knowledge Base Management ---
with st.sidebar:
    st.title("🗂️ Knowledge Base")
    
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF/Text Files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Ingest", type="primary"):
        if uploaded_files:
            with st.spinner("Ingesting documents..."):
                for uploaded_file in uploaded_files:
                    # Streamlit's UploadedFile has a .read() like a normal file
                    try:
                        doc_id, num_chunks = app.ingest_file(uploaded_file, uploaded_file.name)
                        if num_chunks > 0:
                            st.success(f"✅ **{uploaded_file.name}**: Ingested {num_chunks} chunks.")
                        else:
                            st.warning(f"⚠️ **{uploaded_file.name}**: Skipped (Empty or Duplicate).")
                    except Exception as e:
                        st.error(f"❌ **{uploaded_file.name}**: Error - {e}")
        else:
            st.warning("⚠️ No file selected.")
            
    st.markdown("---")
    
    st.markdown("### 📚 Managed Documents")
    if st.button("🔄 Refresh"):
        pass # Streamlit natively reruns on button click, so this updates the list below
        
    try:
        docs = app.list_documents()
        if docs:
            df = pd.DataFrame(docs)[["id", "filename", "created_at"]]
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Delete interface
            doc_to_delete = st.selectbox(
                "Select to Delete", 
                options=[d["id"] for d in docs],
                format_func=lambda x: next(d["filename"] for d in docs if d["id"] == x)
            )
            
            if st.button("🗑️ Delete Selected"):
                try:
                    deleted_name = app.delete_document(doc_to_delete)
                    st.success(f"🗑️ Deleted: {deleted_name}")
                    st.rerun() # Refresh the list immediately
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            st.info("No documents uploaded yet.")
    except Exception as e:
        st.error("Failed to load documents.")

# --- Main Area: Chat Interface ---
st.title("🤖 RAG Research Assistant")

# New Chat Button
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("🧹 New Chat"):
        reset_chat()
        st.rerun()

# Display chat history from state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                result = app.chat(prompt, st.session_state.conversation_id)
                
                raw_msg = result.get("response", "No response received!")
                st.session_state.conversation_id = result.get("conversation_id")
                
                # Format sources nicely
                bot_msg = raw_msg
                if result.get("sources"):
                    sources_list = list(set(s.get("filename", "Unknown") for s in result["sources"]))
                    sources_html = (
                        "\n\n**📚 Sources:**\n" + 
                        "\n".join([f"- {s}" for s in sources_list])
                    )
                    bot_msg += sources_html
                
                st.markdown(bot_msg, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": bot_msg})
                
            except Exception as e:
                error_msg = f"⚠️ **Error:** {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
