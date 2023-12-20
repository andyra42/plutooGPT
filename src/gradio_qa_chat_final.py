# import imports an entire code library. from import imports a specific member or members of the library.
import gc
import logging
import os
import time
# PyTorch is an open source machine learning (ML) framework based on the Python programming language and the Torch
# library. Torch is an open source ML library used for creating deep neural networks and is written in the Lua
# scripting language.
import torch
# Gradio is an open-source Python library that is used to build machine learning and data science demos and web
# applications.
import gradio as gr
# LangChain provides a callbacks system that allows you to hook into the various stages of your LLM application. This
# is useful for logging, monitoring, streaming, and other tasks.
from langchain.callbacks.manager import CallbackManager
# Streaming helps reduce this perceived latency by returning the output of the LLM token by token, instead of all
# at once In the context of a chat application, as a token is generated by the LLM, it can be served immediately
# to the user.
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response

# Chains is an incredibly generic concept which returns to a sequence of modular components (or other chains)
# combined in a particular way to accomplish a common use case. The RetrievalQAChain is a chain that combines a
# Retriever and a QA chain . It is used to retrieve documents from a Retriever and then use a QA
# chain to answer a question based on the retrieved documents.
from langchain.chains import RetrievalQA

# The Hugging Face Inference API allows us to embed a dataset using a quick POST call easily. Since the embeddings
# capture the semantic meaning of the questions, it is possible to compare different embeddings and see how different
# or similar they are
from langchain.embeddings import HuggingFaceInstructEmbeddings
# The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most
# of the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity
# Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering
from langchain.llms import HuggingFacePipeline

# A vector store takes care of storing embedded data and performing vector search
# Chroma is a vector store and embeddings database designed from the ground-up to make it easy to build AI
# applications with embeddings
from langchain.vectorstores import Chroma
# Transformers provides APIs to quickly download and use those pretrained models on a given text,
# fine-tune them on your own datasets and then share them with the community
from transformers import (
    GenerationConfig,
    pipeline,
)
from util.prompt_template_utils import get_prompt_template

from config.constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS
)
from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"

SHOW_SOURCES = True
USE_HISTORY = False
logging.info(f"Running on: {DEVICE_TYPE}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")
logging.info(f"Use history set to: {USE_HISTORY}")
FORMATTED_SOURCES = ""


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


LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)


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
    use_history = False

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    # llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)
    print("USE_HISTORY")
    print(USE_HISTORY)
    # print("prompt" + prompt)
    if USE_HISTORY:
        qa = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa


QA = retrieval_qa_pipline(DEVICE_TYPE, SHOW_SOURCES, promptTemplate_type=LLM)


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
    # qa = retrieval_qa_pipline("cuda", True, promptTemplate_type="llama")

    res = QA(history[-1][0])
    answer, docs = res["result"], res["source_documents"]
    formatted_sources =""
    print("####################DOCS###############")
    print("----------------------------------SOURCE DOCUMENTS---------------------------")
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
        formatted_sources = formatted_sources+document.metadata["source"] + ":"+document.page_content+"</br>"
    print("----------------------------------SOURCE DOCUMENTS---------------------------")
    FORMATTED_SOURCES = formatted_sources
    # history[-1][1] = answer
    history[-1][1] = ""
    for character in answer:
        history[-1][1] += character
        time.sleep(0.01)
        yield history

    return history


def main():
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

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    # Configure gradio QA app
    print("Configuring gradio app")

    with gr.Blocks(gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.purple)) as demo:
        gr.Markdown('''# Application Support \n This Application Utilizes RAG Retrieval-Augmented Generation) an AI 
        framework that allows LLMs to retrieve facts from an external knowledge base to supplement their internal 
        representation of information.''')

        with gr.Row():
            with gr.Column(scale=1, variant='panel', visible=True):
                with gr.Accordion(label="Text generation tuning parameters"):
                    temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.1, step=0.05)
                    max_new_tokens = gr.Slider(label="max_new_tokens", minimum=1, maximum=2048, value=512, step=1)
                    repetition_penalty = gr.Slider(label="repetition_penalty", minimum=0, maximum=2, value=1.1,
                                                   step=0.1)
                    top_k = gr.Slider(label="top_k", minimum=1, maximum=1000, value=10, step=1)
                    top_p = gr.Slider(label="top_p", minimum=0, maximum=1, value=0.95, step=0.05)
                    k_context = gr.Slider(label="k_context", minimum=1, maximum=15, value=5, step=1)
                instruction = gr.Textbox(label="System instruction", lines=3,
                                         value="Use the following pieces of context to answer the question at the end by."
                                               "Generate the answer based on the given context only.If you do not find "
                                               "any information related to the question in the given context, "
                                               "just say that you don't know, don't try to make up an answer. Keep your"
                                               "answer expressive.")
            with gr.Column(scale=3, variant='panel'):
                chatbot = gr.Chatbot([], elem_id="chatbot",
                                     label='Chatbox', height=550, )
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
                with gr.Accordion("Sources"):
                    gr.Markdown(FORMATTED_SOURCES)

        # Launch gradio app
    print("Launching gradio app")
    demo.queue(max_size=10)
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
