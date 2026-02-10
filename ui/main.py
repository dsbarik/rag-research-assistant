import gradio as gr


def chat_response(message, history):
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "Hello djajdfjalk"})
    return "", history


theme = gr.themes.Glass(
    primary_hue=gr.themes.colors.purple
)

css = """
#bot {
    border: 2px solid red !important;
}
"""

with gr.Blocks(theme=theme, css=css) as demo:
    with gr.Row():
        with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=680, elem_id="bot")
                msg = gr.Textbox()
                chat_history = gr.State([])
                
                msg.submit(
                    fn=chat_response,
                    inputs=[msg, chat_history],
                    outputs=[msg, chatbot]
                )

        with gr.Column(scale=1):
            file_inp = gr.File(label="Select Document")
        
if __name__ == "__main__":
    demo.launch()