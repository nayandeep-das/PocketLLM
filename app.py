#!/usr/bin/env python
# coding: utf-8


# In[180]:


import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_classic.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import torch
import inspect
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')


# In[181]:


from transformers import TextIteratorStreamer
from threading import Thread


# In[182]:


import json
import os
import math


# In[183]:


MEMORY_FILE = "chat_memory.json"


# In[184]:


model_path = "models/qwen"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"
)


# In[185]:


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)


# In[186]:


if os.path.exists(MEMORY_FILE):

    with open(MEMORY_FILE, "r") as f:
        chat_history = json.load(f)

else:

    chat_history = []


# In[187]:


from sentence_transformers import SentenceTransformer

pdf_text = ""

pdf_chunks = []

pdf_embeddings = None

faiss_index = None

pdf_mode = False

embedding_model = SentenceTransformer(
    "models/all-MiniLM-L6-v2"
)


# In[188]:


from faster_whisper import WhisperModel

whisper_model = WhisperModel(
    "base",
    download_root="models/whisper",
    device="cpu",
    compute_type="int8"
)


# In[189]:


import piper

from piper import PiperVoice
import wave

voice = PiperVoice.load(
    "models/piper_models/en_US-lessac-medium.onnx"
)

import wave

def text_to_speech(text):

    output_file = "response.wav"

    with wave.open(output_file, "wb") as wav_file:

        voice.synthesize_wav(
            text,
            wav_file
        )

    return output_file


# In[190]:


def voice_chat(audio_file, history):

    text = transcribe_audio(audio_file)

    print("Transcribed Text Length:", len(text))

    if len(text.split()) > 100:

        history.append({
            "role": "assistant",
            "content": "⚠️ Voice input too long. Please keep recordings under 15 seconds."
        })

        yield "", history, None

        return


    history.append({
        "role": "user",
        "content": text
    })

    history.append({
        "role": "assistant",
        "content": ""
    })

    response = ""

    for partial_response in chatbot_response_stream(text):

        response = partial_response

        history[-1]["content"] = partial_response

        yield "", history, None

    audio_file = text_to_speech(response)

    yield "", history, audio_file


# In[191]:


def transcribe_audio(audio_file):

    if audio_file is None:
        return ""

    print("Transcribing...")

    segments, info = whisper_model.transcribe(
        audio_file
    )

    text = ""

    for segment in segments:
        text += segment.text + " "

    print("Transcription Complete")

    return text.strip()


# In[192]:


def save_memory():

    with open(MEMORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=4)



# In[193]:


import faiss

def load_pdf(file):

    global pdf_text
    global pdf_chunks
    global pdf_embeddings
    global faiss_index
    global pdf_mode

    pdf_mode = True

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    pdf_text = text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    pdf_chunks = splitter.split_text(pdf_text)

    print("Creating embeddings...")

    pdf_embeddings = embedding_model.encode(
        pdf_chunks,
        convert_to_numpy=True
    )

    dimension = pdf_embeddings.shape[1]

    faiss_index = faiss.IndexFlatL2(dimension)

    faiss_index.add(pdf_embeddings)

    print("FAISS Index Created")

    return (
    f"""
    ### 📊 PDF STATUS

    ✅ PDF Loaded Successfully

    📄 Characters: {len(pdf_text)}

    🧩 Chunks: {len(pdf_chunks)}

    🔍 Vector Index: Active
    """,

    """
    🔵 PDF Mode Active

    Semantic Search Enabled
    """,

    f"""
    📂 Loaded PDF

    {os.path.basename(file.name)}

    Chunks: {len(pdf_chunks)}

    Vector Index: Active
    """
    )


# In[194]:


def get_relevant_chunks(question, k=5):

    global faiss_index
    global pdf_chunks
    global embedding_model

    if faiss_index is None:
        return []

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = faiss_index.search(
        question_embedding,
        k
    )

    print("Distances:", distances[0])

    retrieved_chunks = []

    for idx in indices[0]:

        if idx < len(pdf_chunks):

            retrieved_chunks.append(
            (idx, pdf_chunks[idx])
        )

    print("Retrieved Chunks:", len(retrieved_chunks))

    return retrieved_chunks


# In[195]:


def chatbot_response_stream(user_input):


    global chat_history
    global pdf_text
    global pdf_chunks

    messages = [
        {
            "role": "system",
            "content": """
        You are PocketLLM, a friendly offline AI assistant running locally on the Qwen2.5-1.5B-Instruct model.

        If asked what model you are using, answer exactly:
        "I am PocketLLM running locally on the Qwen2.5-1.5B-Instruct model."

        Be friendly, helpful, and conversational.

        Use occasional emojis (maximum 2 per response).

        Answer directly and concisely.

        Use bullet points when presenting lists.

        Remember information shared during the conversation.

        Always respond in English unless the user explicitly requests another language.

        Do not claim to be ChatGPT, Claude, Gemini, or any other AI assistant.

        You are designed to run locally on the user's device and do not require an internet connection for normal conversations.
        """
        }
    ]


    if pdf_mode and pdf_chunks:

        print("📄 PDF MODE ACTIVE")

        relevant_chunks = get_relevant_chunks(user_input)

        print("Retrieved Chunks:", len(relevant_chunks))

        if relevant_chunks:

            source_chunks = []

            for idx, chunk in relevant_chunks:
                source_chunks.append(chunk)

            pdf_context = "\n\n---\n\n".join(source_chunks)

            sources = [idx for idx, chunk in relevant_chunks]

            print("Sources:", sources)

            print("\n===== RETRIEVED CONTEXT =====")
            print(pdf_context[:1000])
            print("=============================\n")


            messages = [
            {
                "role": "system",
                "content": f"""
            You are a PDF Question Answering assistant.

            Use ONLY the information contained in the PDF context.

            If the answer is not explicitly stated in the PDF context, reply exactly:

            I could not find that information in the uploaded PDF.

            Never use outside knowledge.
            Never make assumptions.
            Never guess.
            Never invent names, roles, or details.

            PDF Context:

            {pdf_context}
            """
            },
            {
                "role": "user",
                "content": user_input
            }
        ]

        else:

            messages = [
                {
                "role": "system",
                "content": """
            You are a PDF Question Answering assistant.

            No relevant PDF information was found.

            Reply exactly:

            I could not find that information in the uploaded PDF.
            """
            },
            {
                "role": "user",
                "content": user_input
            }
        ]


    else:

        messages.extend(chat_history)

        messages.append({
        "role": "user",
        "content": user_input
    })



    chat_history.append({
        "role": "user",
        "content": user_input
    })

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.5,
        top_p=0.9
    )

    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )

    thread.start()

    response = ""

    for token in streamer:


        for char in token:
            response += char
            yield response


    chat_history.append({
        "role": "assistant",
        "content": response
    })


    MAX_HISTORY = 40

    if len(chat_history) > MAX_HISTORY:
        chat_history[:] = chat_history[-MAX_HISTORY:]

    save_memory()





# In[197]:


def enable_pdf_mode():

    global pdf_mode
    pdf_mode = True

    return """
        🔵 PDF Mode Active

        Vector Search Enabled
        """


def enable_normal_mode():

    global pdf_mode
    pdf_mode = False

    return """
        🟢 Chat Mode Active

        Memory Enabled
        """


# In[198]:


def clear_memory():

    global chat_history

    chat_history = []

    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)

    return [], """
### 🧠 MEMORY

░░░░░░░░░░

0% Used

0 / 100 Messages
"""


# In[199]:


def export_chat():

    os.makedirs("exports", exist_ok=True)

    file_name = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    file_path = os.path.join(
        "exports",
        file_name
    )

    with open(file_path, "w", encoding="utf-8") as f:

        f.write("=" * 50 + "\n")
        f.write("POCKETLLM CHAT EXPORT\n")
        f.write("=" * 50 + "\n\n")

        for msg in chat_history:

            if msg["role"] == "user":
                f.write("[USER]\n")

            elif msg["role"] == "assistant":
                f.write("[POCKETLLM]\n")

            else:
                f.write(f"[{msg['role'].upper()}]\n")

            f.write(msg["content"])
            f.write("\n\n")
            f.write("-" * 50)
            f.write("\n\n")

    return file_path

# In[200]:


def get_memory_status():

    used = math.ceil(len(chat_history) / 2)

    percent = min(used * 2, 100)

    filled = "█" * (percent // 10)
    empty = "░" * (10 - percent // 10)

    return f"""
### 🧠 MEMORY

{filled}{empty}

{percent}% Used

{used} / 100 Conversations
"""


# In[201]:


def chat(message, history):

    history.append({
        "role": "user",
        "content": message
    })

    history.append({
        "role": "assistant",
        "content": ""
    })

    memory_card = get_memory_status()

    yield "", history, memory_card

    for partial_response in chatbot_response_stream(message):

        history[-1]["content"] = partial_response

        memory_card = get_memory_status()

        yield "", history, memory_card


# In[202]:


css = """
/* Background */
.gradio-container{
    background: linear-gradient(
        135deg,
        #031224,
        #071c38,
        #0b2d55
    );
    color: white;
}

/* Title */
h1{
    color:#38bdf8 !important;
    font-size:52px !important;
    font-weight:800 !important;
}

h3{
    color:#cbd5e1 !important;
}

/* Cards */
.gr-box,
.gr-panel{
    background:#111827 !important;
    border:1px solid #1e3a5f !important;
    border-radius:16px !important;
    box-shadow:0 0 15px rgba(56,189,248,0.15);
}

/* Buttons */
button{
    background:linear-gradient(
        90deg,
        #0284c7,
        #38bdf8
    ) !important;

    border:none !important;
    border-radius:12px !important;

    color:white !important;
    font-weight:700 !important;

    transition:0.3s;
}

button:hover{
    transform:translateY(-2px);
    box-shadow:0 0 20px rgba(56,189,248,0.5);
}

/* Chatbot */
.message.user{
    background:#0ea5e9 !important;
    color:white !important;
    border-radius:16px !important;
}

.message.bot{
    background:#1e293b !important;
    color:white !important;
    border-radius:16px !important;
}

/* Textbox */
textarea{
    background:#0f172a !important;
    color:white !important;
    border:1px solid #38bdf8 !important;
}

/* File Upload */
.file-preview{
    background:#0f172a !important;
}

/* Audio Components */
audio{
    border-radius:12px;
}

/* Scrollbar */
::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-thumb{
    background:#38bdf8;
    border-radius:10px;
}

#pdf-status-card{
    background:#0f172a !important;
    border:1px solid #38bdf8 !important;
    border-radius:14px !important;
    padding:12px !important;
    box-shadow:0 0 12px rgba(56,189,248,0.25);
}

#memory-card{
    background:#0f172a !important;
    border:1px solid #22c55e !important;
    border-radius:14px !important;
    padding:12px !important;
    box-shadow:0 0 12px rgba(34,197,94,0.2);
}

#mode-card{
    background:#0f172a !important;
    border:1px solid #0ea5e9 !important;
    border-radius:14px !important;
    padding:12px !important;
    box-shadow:0 0 12px rgba(14,165,233,0.2);
}

textarea::placeholder{
    color:#94a3b8 !important;
}
"""


# In[203]:


with gr.Blocks(css=css) as demo:

    gr.Markdown("""
    # POCKETLLM

    ### Portable Offline AI Assistant
    """)

    with gr.Row():

        # LEFT SIDEBAR
        with gr.Column(scale=1):

            pdf_file = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )

            pdf_status = gr.Markdown(
            """
            ### 📊 PDF STATUS

            📄 No PDF Loaded

            🔍 Vector Index: Inactive
            """,
                elem_id="pdf-status-card"
            )

            pdf_info = gr.Markdown(
            """
            📂 No PDF Loaded
            """
            )

            pdf_mode_status = gr.Markdown(
            """
            🟢 Chat Mode Active
            """,
            elem_id="mode-card"
            )

            memory_status = gr.Markdown(
            """
            ### 🧠 MEMORY

            0 / 100 Messages
            """,
            elem_id="memory-card"
            )

            with gr.Row():

                normal_mode_btn = gr.Button(
                    "💬 Chat"
                )

                pdf_mode_btn = gr.Button(
                    "📄 PDF"
                )

            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Voice Input"
            )

            transcribe_btn = gr.Button(
                "🎙️ Transcribe"
            )

            download_file = gr.File(
                label="Download Export"
            )

            with gr.Row():

                export_btn = gr.Button(
                    "📄 Export"
                )

                clear_btn = gr.Button(
                    "🗑️ Clear"
                )

        # RIGHT SIDE
        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                value=[],
                height=700,

            )

            audio_output = gr.Audio(
                label="🔊 PocketLLM Voice",
                autoplay=True,
                visible=True
            )

            msg = gr.Textbox(
                placeholder="Ask PocketLLM anything...",
                container=False
            )

    msg.submit(
    chat,
    inputs=[msg, chatbot],
    outputs=[
        msg,
        chatbot,
        memory_status
        ]
    )

    clear_btn.click(
    clear_memory,
    outputs=[
        chatbot,
        memory_status
        ]
    )

    export_btn.click(
    export_chat,
    outputs=download_file
    )

    pdf_file.upload(
        load_pdf,
        inputs=pdf_file,
        outputs=[
            pdf_status,
            pdf_mode_status,
            pdf_info
        ]
    )

    normal_mode_btn.click(
    enable_normal_mode,
    outputs=pdf_mode_status
    )

    pdf_mode_btn.click(
    enable_pdf_mode,
    outputs=pdf_mode_status
    )

    transcribe_btn.click(
    voice_chat,
    inputs=[audio_input, chatbot],
    outputs=[msg, chatbot, audio_output]
    )


if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        share=False
    )
