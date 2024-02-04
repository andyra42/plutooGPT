import gradio as gr
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def ask(message, history, schema):

    prompt = """### Instructions:
    Your task is convert a question into a SQL query, given a Postgres database schema.
    Adhere to these rules:
    - **Deliberately go through the question and database schema word by word** to appropriately answer the question
    - **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
    - When creating a ratio, always cast the numerator as float

    ### Input:
    Generate a SQL query that answers the questioq `{question}`.
    This query will run on a database whose schema is represented in this string:`{schema}`
    ### Response:
    Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
    ```sql
    """.format(question=message, schema=schema)
    print("Message : "+message)
    print("prompt"+prompt)

    model_name = "defog/sqlcoder"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # torch_dtype=torch.bfloat16,
        # load_in_8bit=True,
        load_in_4bit=True,
        device_map="auto",
        use_cache=True,
    )
    eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=5
    )

    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return tokenizer.decode(outputs[0].split("```sql")[-1].split("```")[0].split(";")[0].strip() + ";")


# with gr.Blocks() as server:
#     with gr.Tab("LLM Inferencing"):
#         model_input = gr.Textbox(label="Your Question:",
#                                  value="What’s your question?", interactive=True)
#         ask_button = gr.Button("Ask")
#         model_output = gr.Textbox(label="The Answer:", interactive=False,
#                                   value="Answer goes here...")
#
#     ask_button.click(ask, inputs=[model_input], outputs=[model_output])
#
# server.launch()

with gr.Blocks() as demo:
    db_schema = gr.Textbox("You are helpful AI.", label="Enter the database schema", lines=20)
    system_prompt = gr.Textbox(
        " if the question cannot be answered given the database schema, return \"I do not know\"",
        label="System Prompt")

    gr.ChatInterface(
        ask, additional_inputs=[db_schema]
    )

demo.queue().launch(share=True)
