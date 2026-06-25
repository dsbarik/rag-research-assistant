import os
import re
import sys

import gradio as gr
import pandas as pd

sys.path.append(os.getcwd())

from src import ResearchAssistant

app = ResearchAssistant()

# Load custom CSS
with open(os.path.join(os.path.dirname(__file__), "style.css"), "r") as f:
    custom_css = f.read()

# --- Logic Functions ---


def preprocess_latex(text):
    """
    Converts LaTeX delimiters to formats reliably handled by Gradio.
    \\[ ... \\] -> $$ ... $$
    \\( ... \\) -> $ ... $
    """
    if not text:
        return text
    # Replace \[ ... \] with $$ ... $$
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    # Replace \( ... \) with $ ... $
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    return text


def add_user_message(message, history):
    """Updates the UI with the user's message immediately."""
    if not message:
        return gr.update(), history
    history.append({"role": "user", "content": [{"type": "text", "text": message}]})
    return "", history


def get_rag_response(history, conversation_id_state):
    """Queries RAG backend and updates UI."""
    try:
        if not history or history[-1]["role"] != "user":
            return history, conversation_id_state

        last_user_content = history[-1]["content"]

        if isinstance(last_user_content, list):
            last_user_message = last_user_content[0].get("text", "")
        else:
            last_user_message = str(last_user_content)

        result = app.chat(last_user_message, conversation_id_state)

        raw_msg = result.get("response", "No response received!")
        bot_msg = preprocess_latex(raw_msg)
        new_cov_id = result.get("conversation_id")

        # Format sources nicely
        if result.get("sources"):
            sources_list = list(
                set(s.get("filename", "Unknown") for s in result["sources"])
            )
            sources_html = (
                "<div class='source-citation'><strong>📚 Sources:</strong><br>"
                + "<br>".join([f"• {s}" for s in sources_list])
                + "</div>"
            )
            bot_msg += "\n" + sources_html

        history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": bot_msg}],
        })
        return history, new_cov_id

    except Exception as e:
        history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": f"⚠️ **Error:** {str(e)}"}],
        })
        return history, conversation_id_state


def upload_file(files):
    if not files:
        return "⚠️ No file selected."

    file_paths = [files] if not isinstance(files, list) else files
    results = []

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as stream:
                result = app.ingest_file(stream, filename)
                doc_id = result.get("document_id", 0)
                num_chunks = result.get("chunks_processed", 0)
                if num_chunks > 0:
                    results.append(f"✅ **{filename}**: Ingested {num_chunks} chunks.")
                else:
                    results.append(f"⚠️ **{filename}**: Skipped (Empty or Duplicate).")
        except Exception as e:
            results.append(f"❌ **{filename}**: Error - {e}")

    return "\n".join(results)


def get_documents_df():
    try:
        data = app.list_documents()
        if data:
            df = pd.DataFrame(data)
            return df[["id", "filename", "created_at"]]
        return pd.DataFrame(columns=["id", "filename", "created_at"])
    except Exception:
        return pd.DataFrame(columns=["id", "filename", "created_at"])


def get_document_choices():
    try:
        data = app.list_documents()
        return [(f"{d['filename']} (ID: {d['id']})", d["id"]) for d in data]
    except Exception:
        return []


def refresh_documents():
    return get_documents_df(), gr.update(choices=get_document_choices(), value=None)


def delete_selected_document(doc_id):
    if not doc_id:
        return "⚠️ Select a document first.", *refresh_documents()
    try:
        filename = app.delete_document(doc_id)
        return f"🗑️ Deleted: {filename}", *refresh_documents()
    except Exception as e:
        return f"❌ Error: {e}", *refresh_documents()


def reset_conversation():
    return None, []  # Reset conversation_id and history


# --- UI Layout ---

with gr.Blocks(title="RAG Assistant") as demo:
    conversation_id_state = gr.State(value=None)

    with gr.Row():
        # --- Sidebar: Knowledge Management ---
        with gr.Column(scale=1, min_width=300, variant="panel"):
            gr.Markdown("## 🗂️ Knowledge Base", elem_classes=["sidebar-header"])

            with gr.Accordion("📤 Upload Documents", open=True):
                file_input = gr.File(
                    label="Select PDF/Text Files",
                    file_types=[".pdf", ".txt"],
                    file_count="multiple",
                )
                upload_btn = gr.Button(
                    "🚀 Ingest", variant="primary", elem_classes=["primary"]
                )
                upload_status = gr.Markdown()

            gr.Markdown("---")

            with gr.Accordion("📚 Managed Documents", open=True):
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")

                doc_list = gr.Dataframe(
                    value=get_documents_df,
                    headers=["id", "file", "date"],
                    datatype=["number", "str", "str"],
                    interactive=False,
                    elem_classes=["table-wrap"],
                )

                delete_dropdown = gr.Dropdown(
                    label="Select to Delete",
                    choices=get_document_choices(),
                    interactive=True,
                )
                delete_btn = gr.Button(
                    "🗑️ Delete Selected", variant="stop", elem_classes=["stop"]
                )
                delete_status = gr.Markdown()

        # --- Main Area: Chat ---
        with gr.Column(scale=3):
            gr.Markdown("# 🤖 RAG Research Assistant")

            chatbot = gr.Chatbot(
                height=650,
                # type="messages",
                show_label=False,
                avatar_images=(
                    None,
                    "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=rag",
                ),
                render_markdown=True,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": True},
                    {"left": "\\(", "right": "\\)", "display": False},
                    {"left": "\\[", "right": "\\]", "display": True},
                ],
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    container=False,
                    scale=5,
                    autofocus=True,
                )
                submit_btn = gr.Button(
                    "Send", variant="primary", scale=1, min_width=100
                )

            with gr.Row():
                clear_btn = gr.Button("🧹 New Chat", variant="secondary", size="sm")

    # --- Event Wiring ---

    # Chat Events
    msg_submit = msg.submit(
        add_user_message, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        get_rag_response,
        [chatbot, conversation_id_state],
        [chatbot, conversation_id_state],
    )

    submit_btn.click(
        add_user_message, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        get_rag_response,
        [chatbot, conversation_id_state],
        [chatbot, conversation_id_state],
    )

    clear_btn.click(reset_conversation, outputs=[conversation_id_state, chatbot])

    # Document Management Events
    upload_btn.click(upload_file, inputs=[file_input], outputs=[upload_status]).success(
        refresh_documents, outputs=[doc_list, delete_dropdown]
    )

    refresh_btn.click(refresh_documents, outputs=[doc_list, delete_dropdown])

    delete_btn.click(
        delete_selected_document,
        inputs=[delete_dropdown],
        outputs=[delete_status, doc_list, delete_dropdown],
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, theme=gr.themes.Soft(), css=custom_css)
