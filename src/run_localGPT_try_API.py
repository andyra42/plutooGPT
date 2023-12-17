# run_localGPT_API.py
import logging
import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from config.constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import argparse
from flask import Flask, request, jsonify  # Run "pip install flask"
import requests
import json

server_error_code = 503
server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"

g_model_id = g_model_basename = g_webui_port = g_serving_port = g_model_type = ""

LOCALLLMSDAT = "localLLMS.dat"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def post_process_answer(answer, sourceDocs):
    output = []
    for document in sourceDocs:
        output.append("<b>Source</b>: {}<br><b>Content</b><pre>{}</pre>".format(document.metadata["source"],
                                                                                "<br>".join(
                                                                                    document.page_content.split("\n"))))

    answer += f"<br><br>{output}"
    answer = answer.rstrip("\n")
    answer = answer.replace("\n\n", "")
    answer = answer.replace("\n", "")
    answer = answer.replace(" <br><br>", " ")
    answer = answer.replace(" <BR><BR>", " ")
    return answer


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_model(device_type, model_id, model_basename=None):
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

    # model_basename = model_basename.lower()  #lets lowerase the model basename
    model_basename = None if g_model_basename in ("n/a", "") else g_model_basename

    if model_basename is not None:
        model_basename = model_basename.lower()

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            # The code supports all huggingface models that ends with GPTQ and have some variation
            # of .no-act.order or .safetensors in their HF repo.
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif (
            device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

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
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LOCALLLMSDAT = "localLLMS.dat"


def print_header():
    print("........................................................................")
    print("localGPT: Model serving")
    print("........................................................................")
    print("Please choose the SEQ# of the model to serve and the device type and we will be on our way")
    print("...")


def print_table():
    with open(LOCALLLMSDAT, 'r') as file:
        content = file.read().rstrip()
        print(content, end='')


def validate_seq(value):
    while value < 1 or value > 13:
        value = int(input('Please enter a valid SEQ#..Range[1-13]: '))
    return value


print_header()
print_table()
print("\n........................................................................")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_seq():
    try:
        seq = int(input('Please enter the SEQ#..Range[1-13]: '))
        return validate_seq(seq)
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return get_seq()


def get_device_type():
    available_device_types = [
        "cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip",
        "ve", "fpga", "ort", "xla", "lazy", "vulkan", "mps", "meta", "hpu", "mtia",
    ]
    print("Available device types:")
    print(", ".join(available_device_types))
    device_type = input("Device to run on: ")
    if device_type in available_device_types:
        return device_type
    else:
        print("Invalid device type. Please try again.")
        return get_device_type()


parser = argparse.ArgumentParser(description='Model serving options.')
parser.add_argument('--seq', type=int, help='The sequence number to select the model.')
parser.add_argument('--device_type', choices=[
    "cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip",
    "ve", "fpga", "ort", "xla", "lazy", "vulkan", "mps", "meta", "hpu", "mtia",
], help='Device to run on.')

args = parser.parse_args()

if args.seq is None:
    args.seq = get_seq()
else:
    args.seq = validate_seq(args.seq)

if args.device_type is None:
    args.device_type = get_device_type()

seq = args.seq
device_type = args.device_type

print("..................................................")
print(f"Model REST API Server Running on: {device_type}")
print("..................................................")


def getModelDesignInfo(seq):
    with open(LOCALLLMSDAT, 'r') as file:
        lines = file.readlines()
        for line in lines[3:]:  # Skip the header
            columns = line.strip().split('|')
            if int(columns[0]) == seq:
                # g_webui_port = columns[1]
                # g_serving_port = columns[2]
                # g_model_type = columns[3]
                # g_model_id = columns[4]
                # g_model_basename = columns[5]
                # print(f"Selected SEQ#: {seq}")
                # print(f"WEBUI PORT: {g_webui_port}")
                # print(f"SERVING PORT: {g_serving_port}")
                # print(f"MODEL TYPE: {g_model_type}")
                # print(f"MODEL ID: {g_model_id}")
                # print(f"MODEL BASENAME: {g_model_basename}")
                break
    return columns[1], columns[2], columns[3], columns[4], columns[5]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
This is the main block of code that implements the information retrieval task.

1. Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
2. Loads the existing vectorestore that was created by inget.py
3. Loads the local LLM using load_model function - You can now set different LLMs.
4. Setup the Question Answer retreival chain.
5. Question answers.
"""

logging.info(f"Running on: {device_type}")
# logging.info(f"Display Source Documents set to: {show_sources}")

embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})

# uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# load the vectorstore
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS,
)
retriever = db.as_retriever()

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
just say that you don't know, don't try to make up an answer.

{context}

{history}
Question: {question}
Helpful Answer:"""

prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
memory = ConversationBufferMemory(input_key="question", memory_key="history")

g_webui_port, g_serving_port, g_model_type, g_model_id, g_model_basename = getModelDesignInfo(seq)

print(f"model_basename: {g_model_basename}")

# llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
# llm = load_model(device_type, model_id=g_model_id, model_basename=g_model_basename)
llm = load_model(device_type, model_id=g_model_id,
                 model_basename=None if g_model_basename in ("n/a", "") else g_model_basename)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "memory": memory},
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create a Flask app
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:

        question = request.json['question']
        system_content = request.json['system_content']
        chatbot = request.json['chatbot']
        history = request.json['history']

        print(f"in /predict; question: {question}")

        # Get the answer from the chain
        prompt = system_content + f"\n Question: {question}"
        res = qa(prompt)
        answer, docs = res['result'], res['source_documents']

        print(f"in /predict; answer: ")
        print(res['result']);

        answer = post_process_answer(answer, docs)
        history.append(question)
        history.append(answer)
        chatbot = [(history[i], history[i + 1]) for i in range(0, len(history), 2)]

        response = jsonify({"chatbot": chatbot, "history": history})

        # Set the headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers["Content-Type"] = "application/json"

        if not question or not system_content:
            raise ValueError("Missing question or system_content parameters")

        return jsonify({"chatbot": chatbot, "history": history}), 200, {"Access-Control-Allow-Origin": "*"}
    except Exception as e:
        print(e)
        chatbot = []
        history = []
        history.append("")
        answer = server_error_msg + f" (error_code: {server_error_code})"
        history.append(answer)
        chatbot = [(history[i], history[i + 1]) for i in range(0, len(history), 2)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        response = jsonify({"chatbot": chatbot, "history": history})
        return jsonify({"chatbot": chatbot, "history": history}), server_error_code, {
            "Access-Control-Allow-Origin": "*"}

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def main(seq, device_type):
    # app.run()
    app.run(host='0.0.0.0', port=g_serving_port)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main(seq, device_type)
