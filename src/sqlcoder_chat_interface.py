import gradio as gr
import time


def echo(message, history, db_schema, system_prompt, tokens):
    response = f"System prompt: {system_prompt}\n Message: {message}."
    for i in range(min(len(response), int(tokens))):
        time.sleep(0.05)
        yield response[: i + 1]


with gr.Blocks() as demo:
    db_schema = gr.Textbox("You are helpful AI.", label="Enter the database schema", lines=20)
    system_prompt = gr.Textbox(
        " if the question cannot be answered given the database schema, return \"I do not know\"",
        label="System Prompt")
    slider = gr.Slider(10, 100, render=False)

    gr.ChatInterface(
        echo, additional_inputs=[db_schema, system_prompt, slider]
    )

demo.queue().launch()
