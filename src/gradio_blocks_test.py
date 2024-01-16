import gradio as gr
import gc, torch
import time

GLOBAL_VAR = "LLAMA"
TEMPARATURE = 0.2


def update(name):
    return f"Welcome to Gradio, {name}!"


def add_text(history, text):
    history = history + [[text, None]]
    print("history")
    print(history)
    return history, ""


def post_process_answer(answer, source):
    answer += f"<br><br>Source: {source}"
    answer = answer.replace("\n", "<br>")
    return answer


def srcshowfn(chkbox):
    vis = True if chkbox == True else False
    print(vis)
    return gr.Textbox.update(visible=vis)


def bot(history,
        instruction="Use the following pieces of context to answer the question at the end. Generate the answer based "
                    "on the given context only if you find the answer in the context. If you do not find any "
                    "information related to the question in the given context, just say that you don't know, "
                    "don't try to make up an answer. Keep your answer expressive.",
        temperature=0.1,
        max_new_tokens=512,
        repetition_penalty=1.1,
        top_k=10,
        top_p=0.95,
        k_context=5,
        num_return_sequences=1,
        ):
    accordion_html = ""
    accordion_html += f"<p>\"main_answer\"</p><hr><br><br><br><br>"
    accordion_item = f"""
            <details>
                <summary><strong>Source 1: test.pdf</strong></summary>
                <pre style='background-color:#d3d3d3;'>source content </pre>
            </details>
            """
    accordion_html1 = "\n\n<b>Hi anand</b>'</br>Rapolu"
    accordion_html += accordion_item
    print("input")
    print(history)
    print(k_context)
    print(top_p)
    print(history[-1][0])
    processed_responses = [[history, accordion_html]]
    history[-1][1] = ""
    for character in accordion_html1:
        history[-1][1] += character
        time.sleep(0.01)
        yield history
    print(history)
    print(gr.HTML(history).value)
    return gr.HTML(history).value


def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()
    return None


with gr.Blocks(gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.purple)) as demo:
    gr.Markdown('''# Application Support \n This Application Utilizes RAG Retrieval-Augmented Generation) an AI 
          framework that allows LLMs to retrieve facts from an external knowledge base to supplement their internal 
          representation of information.''')

    with gr.Row():
        with gr.Column(scale=1, variant='panel'):
            with gr.Accordion(label="Text generation tuning parameters"):
                temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=TEMPARATURE, step=0.05,
                                        interactive=False)
                max_new_tokens = gr.Slider(label="max_new_tokens", minimum=1, maximum=2048, value=512, step=1,
                                           interactive=False)
                repetition_penalty = gr.Slider(label="repetition_penalty", minimum=0, maximum=2, value=1.1, step=0.1)
                top_k = gr.Slider(label="top_k", minimum=1, maximum=1000, value=10, step=1)
                top_p = gr.Slider(label="top_p", minimum=0, maximum=1, value=0.95, step=0.05)
                k_context = gr.Slider(label="k_context", minimum=1, maximum=15, value=5, step=1)
                html = gr.HTML(label="HTML preview", show_label=True)

            instruction = gr.Textbox(label="System instruction", lines=3,
                                     value="Use the following pieces of context to answer the question at the end by. "
                                           "Generate the answer based on the given context only.If you do not find "
                                           "any information related to the question in the given context, "
                                           "just say that you don't know, don't try to make up an answer. Keep your "
                                           "answer expressive.")
            with gr.Accordion("Sources"):
                gr.Markdown("Use the following pieces of context to answer the question at the end by. "
                            "Generate the answer based on the given context only.If you do not find "
                            "any information related to the question in the given context, "
                            "just say that you don't know, don't try to make up an answer. Keep your "
                            "answer expressive.")
                gr.Label(label="LLM MODEL", value=GLOBAL_VAR)
                gr.HTML(label="LLM MODEL", value="<p style=\"color:red;\">Red paragraph text</p>")
                gr.HTML(label="LLM MODEL",
                        value="<b>LLM MODEL</b>" + " :" + "<p style=\"color:red;\">" + GLOBAL_VAR + "</p>",
                        show_label=True)

        with gr.Column(scale=3, variant='panel'):
            chatbot = gr.Chatbot([], elem_id="chatbot",
                                 label='Chatbox', height=600)
            txt = gr.Textbox(label="Question", lines=2, placeholder="Enter your question and press shift+enter ")

            with gr.Row():
                with gr.Column(scale=1):
                    submit_btn = gr.Button('Submit', variant='primary', size='sm')

                with gr.Column(scale=1):
                    clear_btn = gr.Button('Clear', variant='stop', size='sm')
            with gr.Row():
                gr.Examples(
                    [["What is Pool Net ?"], ["What is Trade Comparison in RTTM MBS ?"],
                     ["What is swap explain in detail"]],
                    [txt],
                    chatbot,
                    # cache_examples=True,
                )
                gr.Label(label="LLM MODEL", value=GLOBAL_VAR)
            with gr.Row():
                srcshow = gr.Checkbox(value=False, label='Show sources')
            with gr.Row():
                outputsrc = gr.Textbox(label="Sources", visible=False)
            srcshow.change(srcshowfn, inputs=srcshow, outputs=outputsrc)

            txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot, [chatbot, instruction, temperature, max_new_tokens, repetition_penalty, top_k, top_p, k_context],
                chatbot)
            submit_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot, [chatbot, instruction, temperature, max_new_tokens, repetition_penalty, top_k, top_p, k_context],
                chatbot).then(
                clear_cuda_cache, None, None
            )

demo.launch(enable_queue=True)
