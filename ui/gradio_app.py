import os
import sys

import gradio as gr
import pandas as pd

sys.path.append(os.getcwd())

from src import ResearchAssistant

app = ResearchAssistant()

# --- STEP 1: Add User Message to Chat immediately ---
def add_user_message(message, history):
    """
    Updates the UI with the user's message immediately.
    Returns: cleared text box, updated history
    """
    if not message:
        # If no message, do nothing, just return current state
        return gr.update(), history

    history.append({"role": "user", "content": message})
    return "", history


# --- STEP 2: Process RAG Response ---
def get_rag_response(history, conversation_id_state):
    """
    Uses the last message in history to query the RAG backend.
    Updates the UI with the bot's response when ready.
    """
    try:
        # Get the last user message from the history
        if not history or history[-1]["role"] != "user":
            return history, conversation_id_state
            
        last_user_message = history[-1]["content"]

        # Call the backend
        result = app.chat(last_user_message, conversation_id_state)
        
        bot_msg = result.get("response", "No response received!")
        new_cov_id = result.get("conversation_id")

        # Format sources if they exist
        if result.get("sources"):
            sources_names = set(s.get("filename", "Unknown") for s in result["sources"])
            bot_msg += "\n\n**Sources:** " + ", ".join(sources_names)

        history.append({"role": "assistant", "content": bot_msg})
        
        return history, new_cov_id

    except Exception as e:
        gr.Warning(f"Chat Error: {str(e)}")
        history.append({
            "role": "assistant", 
            "content": "⚠️ I encountered an error processing your request."
        })
        return history, conversation_id_state


# ... [Keep upload_file, get_documents, delete_selected_document as is] ...

def upload_file(files):
    if not files:
        return "⚠️ No file selected."
    file_paths = [files] if not isinstance(files, list) else files
    results = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as stream:
                doc_id, num_chunks = app.ingest_file(stream, filename)
                if num_chunks > 0:
                    results.append(f"✅ File {filename} uploaded successfully.")
                else:
                    results.append(f"⚠️ Duplicate file: {filename}")
        except Exception as e:
            results.append(f"❌ Error uploading {filename}: {e}")
    return "\n".join(results)

def get_documents():
    try:
        data = app.list_documents()
        if data:
            df = pd.DataFrame(data)
            df = df[["id", "filename", "created_at"]]
        else:
            df = pd.DataFrame(columns=["id", "filename", "created_at"])
        choices = [(f"{d['filename']}", d["id"]) for d in data]
        return df, gr.update(choices=choices, value=None)
    except Exception as e:
        print(f"Fetch error: {e}")
        return pd.DataFrame(columns=["id", "filename", "created_at"]), gr.update(choices=[], value=None)

def delete_selected_document(doc_id):
    if not doc_id:
        return "⚠️ Please select a document to delete.", *get_documents()
    try:
        filename = app.delete_document(doc_id)
        return f"✅ Document deleted: {filename}", *get_documents()
    except Exception as e:
        return f"⚠️ Connection Error: {e}", *get_documents()


# --- UI LAYOUT ---

with gr.Blocks(title="RAG Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 RAG Research Assistant")

    conversation_id_state = gr.State(value=None)

    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(height=500, type="messages", label="Conversation")
            msg = gr.Textbox(
                placeholder="Ask a question about your documents...",
                container=False,
                scale=7,
            )
            with gr.Row():
                clear = gr.ClearButton([msg, chatbot], value="Clear Conversation")

            # --- UPDATED EVENT WIRING ---
            
            # 1. When user submits, IMMEDIATELY update UI with user message and clear textbox
            user_submit = msg.submit(
                add_user_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=False, # Don't wait for other processes for this simple UI update
                show_progress="hidden"
            )
            
            # 2. THEN, trigger the RAG bot response
            user_submit.then(
                get_rag_response,
                inputs=[chatbot, conversation_id_state],
                outputs=[chatbot, conversation_id_state],
                show_progress="hidden"
            )

            # Reset conversation ID when clearing chat
            clear.click(lambda: None, outputs=[conversation_id_state])

        # TAB 2: KNOWLEDGE BASE (Same as before)
        with gr.Tab("📂 Knowledge Base"):
            gr.Markdown("### Manage Your Documents")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 1. Upload New PDF")
                    file_input = gr.File(label="Select PDF Files", file_types=[".pdf"], file_count="multiple")
                    upload_btn = gr.Button("🚀 Ingest Document", variant="primary")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)

                with gr.Column(scale=2):
                    gr.Markdown("#### 2. Current Documents")
                    refresh_btn = gr.Button("🔄 Refresh List", size="sm")
                    doc_table = gr.Dataframe(headers=["id", "filename", "created_at"], datatype=["number", "str", "str"], label="Document Index", interactive=False)
                    gr.Markdown("#### 3. Remove Document")
                    with gr.Row():
                        delete_dropdown = gr.Dropdown(label="Select Document to Delete", choices=[], interactive=True, scale=3)
                        delete_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
                    delete_status = gr.Textbox(label="Deletion Status", interactive=False)

    upload_btn.click(upload_file, inputs=[file_input], outputs=[upload_status]).success(
        get_documents, outputs=[doc_table, delete_dropdown]
    )
    refresh_btn.click(get_documents, outputs=[doc_table, delete_dropdown])
    delete_btn.click(delete_selected_document, inputs=[delete_dropdown], outputs=[delete_status, doc_table, delete_dropdown])
    demo.load(get_documents, outputs=[doc_table, delete_dropdown])

if __name__ == "__main__":
    demo.launch(server_port=7860, show_api=False)