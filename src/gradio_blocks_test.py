import gradio as gr
import gc, torch


def update(name):
    return f"Welcome to Gradio, {name}!"


def add_text(history, text):
    history = history + [[text, None]]
    print("history")
    print(history)
    return history, ""


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
    print("input")
    print(history)
    print(k_context)
    print(top_p)
    print(history[-1][0])
    return history


def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()
    return None


with gr.Blocks(gr.themes.Soft(primary_hue=gr.themes.colors.slate, secondary_hue=gr.themes.colors.purple)) as demo:
    gr.Markdown('''# Retrieval Augmented Generation \n
                       RAG (Retrieval-Augmented Generation) addresses the data freshness problem in Large Language Models (LLMs) like Llama-2, which lack awareness of recent events. LLMs perceive the world only through their training data, leading to challenges when needing up-to-date information or specific datasets. To tackle this, retrieval augmentation is employed, enabling relevant external knowledge from a knowledge base to be incorporated into LLM responses.
                       RAG involves creating a knowledge base containing two types of knowledge: parametric knowledge from LLM training and source knowledge from external input. Data for the knowledge base is derived from datasets relevant to the use case, which are then processed into smaller chunks to enhance relevance and efficiency. Token embeddings, generated using models like RoBERTa, are crucial for retrieving context and meaning from the knowledge base.
                       A vector database could be used to manage and search through the embeddings efficiently. The LangChain library facilitates interactions with the knowledge base, allowing LLMs to generate responses based on retrieved information. Generative Question Answering (GQA) or Retrieval Augmented Generation (RAG) techniques instruct the LLM to craft answers using knowledge base content. To enhance trust, answers can be accompanied by citations indicating the information source.
                       RAG leverages a combination of external knowledge and LLM capabilities to provide accurate, up-to-date, and well-grounded responses. This approach is gaining traction in products such as AI search engines and conversational agents, highlighting the synergy between LLMs and robust knowledge bases.
                  ''')

    with gr.Row():
        with gr.Column(scale=1, variant='panel'):
            with gr.Accordion(label="Text generation tuning parameters"):
                temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.1, step=0.05)
                max_new_tokens = gr.Slider(label="max_new_tokens", minimum=1, maximum=2048, value=512, step=1)
                repetition_penalty = gr.Slider(label="repetition_penalty", minimum=0, maximum=2, value=1.1, step=0.1)
                top_k = gr.Slider(label="top_k", minimum=1, maximum=1000, value=10, step=1)
                top_p = gr.Slider(label="top_p", minimum=0, maximum=1, value=0.95, step=0.05)
                k_context = gr.Slider(label="k_context", minimum=1, maximum=15, value=5, step=1)
            instruction = gr.Textbox(label="System instruction", lines=3,
                                     value="Use the following pieces of context to answer the question at the end by. "
                                           "Generate the answer based on the given context only.If you do not find "
                                           "any information related to the question in the given context, "
                                           "just say that you don't know, don't try to make up an answer. Keep your "
                                           "answer expressive.")
        with gr.Column(scale=3, variant='panel'):
            chatbot = gr.Chatbot([], elem_id="chatbot",
                                 label='Chatbox', height=725, )
            txt = gr.Textbox(label="Question", lines=2, placeholder="Enter your question and press shift+enter ")

            with gr.Row():
                with gr.Column(scale=1):
                    submit_btn = gr.Button('Submit', variant='primary', size='sm')

                with gr.Column(scale=1):
                    clear_btn = gr.Button('Clear', variant='stop', size='sm')
            txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot, [chatbot, instruction, temperature, max_new_tokens, repetition_penalty, top_k, top_p, k_context],
                chatbot)
            submit_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot, [chatbot, instruction, temperature, max_new_tokens, repetition_penalty, top_k, top_p, k_context],
                chatbot).then(
                clear_cuda_cache, None, None
            )

demo.launch()
