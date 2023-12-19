import gc
import logging
import os
import torch

import click
import gradio as gr
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from util.prompt_template_utils import get_prompt_template

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)
from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)
from config.constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS
)


def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


def add_text(history, text):
    history = history + [[text, None]]
    print("history")
    print(history)
    return history, ""


def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()
    return None


def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama"):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})
    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama",
    type=click.Choice(
        ["llama", "mistral", "non_llama"],
    ),
    help="model type, llama, mistral or non_llama",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)
def get_responses(question, device_type, use_history, model_type):
    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)

    res = qa(question)
    answer, docs = res["result"], res["source_documents"]

    return answer, docs


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
    return history


def bot1(history, device_type, use_history, model_type,
         ):
    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)

    res = qa(history[-1][0])
    answer, docs = res["result"], res["source_documents"]

    return answer


def main(device_type="cuda", show_sources=True, use_history=False, model_type="llama", save_qa=False):
    """
        Implements the main information retrieval task for a localGPT.

        This function sets up the QA system by loading the necessary embeddings, vectorstore, and LLM model.
        It then enters an interactive loop where the user can input queries and receive answers. Optionally,
        the source documents used to derive the answers can also be displayed.

        Parameters:
        - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'mps', 'cuda', etc.
        - show_sources (bool): Flag to determine whether to display the source documents used for answering.
        - use_history (bool): Flag to determine whether to use chat history or not.

        Notes:
        - Logging information includes the device type, whether source documents are displayed, and the use of history.
        - If the models directory does not exist, it creates a new one to store models.
        - The user can exit the interactive loop by entering "exit".
        - The source documents are displayed if the show_sources flag is set to True.

        """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    # Configure gradio QA app
    print("Configuring gradio app")

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
                    repetition_penalty = gr.Slider(label="repetition_penalty", minimum=0, maximum=2, value=1.1,
                                                   step=0.1)
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
                    bot,
                    [chatbot, instruction, temperature, max_new_tokens, repetition_penalty, top_k, top_p, k_context],
                    chatbot)
                submit_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
                    bot,
                    [chatbot, instruction, temperature, max_new_tokens, repetition_penalty, top_k, top_p, k_context],
                    chatbot).then(
                    clear_cuda_cache, None, None
                )

        # Launch gradio app
    print("Launching gradio app")
    demo.queue(concurrency_count=3)
    demo.launch(share=True,
                enable_queue=True,
                debug=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=7111)
    print("Gradio app ready")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
