# run_localGPT_try_WEBUI.py
import \
    gradio as gr  # import gradio; Remember to do this command from the OS prompt before executing this: "pip3 install gradio"

from config.constants import PERSIST_DIRECTORY

# For predict fnction to call serving model
import requests
import json

server_error_code = 503
server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def clear_history(request: gr.Request):
    state = None
    return ([], state, "")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sourceInterface(docs: dict):
    output = []
    for document in docs:
        output.append("<b>Source</b>: {}<br><b>Content</b><pre>{}</pre>".format(document.metadata["source"],
                                                                                "<br>".join(
                                                                                    document.page_content.split("\n"))))
    return output


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LOCALLLMSDAT = "localLLMS.dat"


def print_header():
    print("........................................................................")
    print("localGPT: Model serving")
    print("........................................................................")
    print("Please choose the SEQ# of the model that is already running/serving")
    print("...")


def print_table():
    with open(LOCALLLMSDAT, 'r') as file:
        content = file.read().rstrip()
        print(content, end='')


def validate_seq(value):
    while value < 1 or value > 13:
        value = int(input('Please enter a valid SEQ#..Range[1-13]: '))
    return value


def get_seq():
    try:
        seq = int(input('Please enter the SEQ#..Range[1-13]: '))
        return validate_seq(seq)
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return get_seq()


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


def get_PORT_INFO(seq):
    webui_port, LLM_SERVING_PORT, _, _, _ = getModelDesignInfo(seq)
    return webui_port, LLM_SERVING_PORT


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import socket


def is_port_in_use(port):
    if isinstance(port, str):  # Check if the port is a string
        port = int(port)  # Convert the port to an integer if it's a string
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
        except socket.error:
            return 1
        return 0


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def process_chatbot_responses1(response_json):
    chatbot_responses = response_json["chatbot"]
    processed_responses = []
    answer = None

    for response in chatbot_responses:
        question, response_text = response
        if answer is None:
            answer = response_text.split("<br><br>")[0]

        response_parts = response_text.split("<br><br>")
        processed_response = response_parts[0]

        # Process the sources and content
        sources_content = response_parts[1:]
        for source_content in sources_content:
            source_content = source_content.replace("<br>", "\n")
            source_content = source_content.replace("['<b>Source</b>:", "\n\n<br><hr><b>Source:</b>")
            source_content = source_content.replace("', '<b>Source</b>:", "\n\n<br><hr><b>Source:</b>")
            source_content = source_content.replace("<b>Content</b><pre>",
                                                    "\n\n<br><b>Content</b>\n<pre style='background-color:#d3d3d3;'>")
            source_content = source_content.replace("</pre>", "</pre>\n")
            processed_response += source_content

        # print("Processed response")
        # print(processed_response)
        # replace both the newline character (\n) and the carriage return character (\r) with a space.
        processed_response = processed_response.replace('\n', ' ').replace('\r', ' ')
        # Remove any trailing characters
        processed_response = processed_response.rstrip(", ']")

        processed_responses.append([question, processed_response])

    return processed_responses, answer


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_html_accordion(processed_response):
    # Split the response using the <hr> delimiter
    sources = processed_response.split("<br><hr>")
    accordion_html = ""

    # Extract the main answer
    main_answer = sources[0]
    accordion_html += f"<p>{main_answer}</p><hr><br><br><br><br>"

    # Iterate over each source
    for idx, source in enumerate(sources[1:], 1):  # Skip the first part as it's the main answer
        # Extract the source title and content
        source_title = source.split("<b>Source:</b>")[1].split("<br><b>Content</b>")[0].strip()
        source_content = source.split("<pre style='background-color:#d3d3d3;'>")[1].split("</pre>")[0].strip()

        # Create the accordion item in HTML format
        accordion_item = f"""
        <details>
            <summary><strong>Source {idx}: {source_title}</strong></summary>
            <pre style='background-color:#d3d3d3;'>{source_content}</pre>
        </details>
        """
        accordion_html += accordion_item

    return accordion_html


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def process_chatbot_responses(response_json):
    chatbot_responses = response_json["chatbot"]
    processed_responses = []
    answer = None

    for response in chatbot_responses:
        question, response_text = response
        if answer is None:
            answer = response_text.split("<br><br>")[0]

        response_parts = response_text.split("<br><br>")
        processed_response = response_parts[0]

        # Process the sources and content
        sources_content = response_parts[1:]
        for source_content in sources_content:
            source_content = source_content.replace("<br>", "\n")
            source_content = source_content.replace("['<b>Source</b>:", "\n\n<br><hr><b>Source:</b>")
            source_content = source_content.replace("', '<b>Source</b>:", "\n\n<br><hr><b>Source:</b>")
            source_content = source_content.replace("<b>Content</b><pre>",
                                                    "\n\n<br><b>Content</b>\n<pre style='background-color:#d3d3d3;'>")
            source_content = source_content.replace("</pre>", "</pre>\n")
            processed_response += source_content

        # print("Processed response")
        # print(processed_response)
        # replace both the newline character (\n) and the carriage return character (\r) with a space.
        processed_response = processed_response.replace('\n', ' ').replace('\r', ' ')
        # Remove any trailing characters
        processed_response = processed_response.rstrip(", ']")
        # print("processed_response: " + processed_response)
        accordion_html = create_html_accordion(processed_response)
        # processed_responses.append([question, processed_response])
        processed_responses.append([question, accordion_html])

    return processed_responses, answer


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def predict(
        model_id: str,
        model_basename: str,
        model_serving_port_number: str,
        question: str,
        system_content: str,
        chatbot: list,
        history: list,
):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    predictURL = f"http://localhost:{model_serving_port_number}/predict"  # This asumes that the model server is running on localhost:model_serving-port_number

    print(
        f"Inside predict function of run_localGPT_try_WEBUI.py. Model id: {model_id} and model basename: {model_basename}")

    headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*"
    }

    data = {
        "question": question,
        "system_content": system_content,
        "chatbot": chatbot,
        "history": history
    }

    json_data = json.dumps(data)

    try:
        response = requests.post(predictURL, headers=headers, data=json_data)

        if response.status_code == 200:
            response_json = response.json()
            # print("Data type of response_json:", type(response_json))
            # print(response_json)
            processed_chatbot_responses, answer = process_chatbot_responses(
                response_json)  # Pass the entire response_json
            history = response_json["history"]
            history.append(question)
            history.append(answer)
            return processed_chatbot_responses, history

        else:
            print("Error:", response.status_code)
            history.append("")
            answer = server_error_msg + f" (error_code: {server_error_code})"
            history.append(answer)
            chatbot = [(history[i], history[i + 1]) for i in range(0, len(history), 2)]
            return chatbot, history

    except Exception as e:
        print(e)
        history.append("")
        answer = server_error_msg + f" (error_code: {server_error_code})"
        history.append(answer)
        chatbot = [(history[i], history[i + 1]) for i in range(0, len(history), 2)]
        return chatbot, history


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def reset_textbox():
    return gr.update(value="")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def predict_wrapper(model_id, model_basename, model_serving_port_number, question, system_content, chatbot, state):
    return predict(model_id=model_id, model_basename=model_basename,
                   model_serving_port_number=model_serving_port_number, question=question,
                   system_content=system_content, chatbot=chatbot, history=state)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main(args):
    title = """<h1 align="center">Web UI Chat with localGPT 🤖</h1>"""
    print(f"Loading the Chatbot Web UI...")
    print(f"..................................................")

    with gr.Blocks(
            css="""
        footer .svelte-1lyswbr {display: none !important;}
        #col_container {margin-left: auto; margin-right: auto;}
        #chatbot .wrap.svelte-13f7djk {height: 120vh; max-height: 120vh}
        #chatbot .message.user.svelte-13f7djk.svelte-13f7djk {width:fit-content; background:orange; border-bottom-right-radius:0}
        #chatbot .message.bot.svelte-13f7djk.svelte-13f7djk {width:fit-content; padding-left: 16px; border-bottom-left-radius:0}
        #chatbot .pre {border:2px solid white;}
        .contain { display: flex; flex-direction: column; }
        .gradio-container { height: 100vh !important; }
        #component-0 { height: 100%; }
        #chatbot { flex-grow: 1; overflow: auto;}
        pre {
        white-space: pre-wrap;       /* Since CSS 2.1 */
        white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
        white-space: -pre-wrap;      /* Opera 4-6 */
        white-space: -o-pre-wrap;    /* Opera 7 */
        word-wrap: break-word;       /* Internet Explorer 5.5+ */
        }
        """
    ) as demo:
        gr.HTML(title)
        with gr.Row():
            g_webui_port, g_serving_port, g_model_type, g_model_id, g_model_basename = getModelDesignInfo(args.seq)
            with gr.Column(elem_id="col_container", scale=0.3):  # 1st column of 2 columns
                with gr.Accordion("System", open=True):
                    system_content = gr.Textbox(
                        value=f"You are accessing the localGPT which has been built with LangChain and {g_model_id} model being served at port# {g_serving_port}",
                        show_label=False)
                with gr.Accordion("Configuration", open=True):
                    _ = gr.Text(value=g_model_type, label="Model Type")
                    _ = gr.Text(value=g_model_id, label="Model ID")
                    _ = gr.Text(value=g_model_basename, label="Model Basename")
                    _ = gr.Text(value=g_webui_port, label="Web UI Port")
                    _ = gr.Text(value=g_serving_port, label="Model Serving Port")
                    _ = gr.Text(value=PERSIST_DIRECTORY, label="Data Persisted Directory")

            with gr.Column(elem_id="col_container"):  # 2nd column of 2 columns
                chatbot = gr.Chatbot(elem_id="chatbot", label="localGPT's Chat Experience")
                question = gr.Textbox(placeholder="Ask something", show_label=False, value="")
                state = gr.State([])
                with gr.Row():
                    with gr.Column():
                        submit_btn = gr.Button(value="🚀 Send")
                    with gr.Column():
                        clear_btn = gr.Button(value="🗑️ Clear history")

        question.submit(
            lambda question, system_content, chatbot, state: predict_wrapper(g_model_id, g_model_basename,
                                                                             g_serving_port, question, system_content,
                                                                             chatbot, state),
            [question, system_content, chatbot, state],
            [chatbot, state],
        )

        submit_btn.click(
            lambda question, system_content, chatbot, state: predict_wrapper(g_model_id, g_model_basename,
                                                                             g_serving_port, question, system_content,
                                                                             chatbot, state),
            [question, system_content, chatbot, state],
            [chatbot, state],
        )

        submit_btn.click(reset_textbox, [], [question])
        clear_btn.click(clear_history, None, [chatbot, state, question])
        question.submit(reset_textbox, [], [question])
        demo.queue(concurrency_count=10, status_update_rate="auto")
        demo.launch(server_name=args.webui_server_name, server_port=args.webui_server_port, share=args.share,
                    debug=args.debug)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    print_header()
    print_table()
    print("\n........................................................................")

    import argparse

    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Web UI serving for selected Large Languange Model (LLM)')

    parser.add_argument("--webui_server_name", default="0.0.0.0")
    parser.add_argument('--webui_server_port', type=int,
                        help='The port for the web UI server.')  # gets reasigned below. please do not assign value here because it will be overwritten
    parser.add_argument("--share", action="store_true",
                        default=True)  # Running on public URL: https://9214f72a5ac010750b.gradio.live  (example url only; Your URL may vary)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--seq', type=int, help='The sequence number to select the model.')

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.seq is None:
        args.seq = get_seq()
    else:
        args.seq = validate_seq(args.seq)

    args.webui_server_port, llm_serving_port, l_model_type, l_model_id, l_model_basename = getModelDesignInfo(args.seq)

    args.webui_server_port = int(args.webui_server_port)
    llm_serving_port = int(llm_serving_port)

    if not is_port_in_use(llm_serving_port):
        print(f"The model needs to be serving on port {llm_serving_port}.")
        print(
            f"Can't possibility continue. Please bring up the model server for MODEL TYPE: {l_model_type}, MODEL ID: {l_model_id} with MODEL_BASENAME {l_model_basename} and invoke this program again")
        exit()

    print(f"..................................................")
    print(f"The localGPT Web UI program is starting now")
    print(f"..................................................")

    main(args)
