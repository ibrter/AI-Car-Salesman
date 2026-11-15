import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import chromadb
from openai import OpenAI
from gtts import gTTS
from dotenv import load_dotenv
import io

# -----------------------------
# 1. Configuration
# -----------------------------

load_dotenv()

st.set_page_config(
    page_title="AI Car Salesman",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 AI Car Salesman")
st.caption("Ask me anything about cars. I'll help you find the right one!")

# -----------------------------
# 2. OpenRouter Setup
# -----------------------------

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    st.error("Please configure OPENROUTER_API_KEY in your .env file.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)

# -----------------------------
# 3. ChromaDB Setup
# -----------------------------

@st.cache_resource
def load_database():
    chroma_client = chromadb.PersistentClient(
        path="./cars_db"
    )
    return chroma_client.get_collection("car_data")

try:
    col = load_database()
except Exception as e:
    st.error(f"Could not load ChromaDB: {e}")
    st.stop()

# -----------------------------
# 4. Chat Function
# -----------------------------

def chat_with_cars(query):

    # Retrieve relevant documents
    results = col.query(
        query_texts=[query],
        n_results=3
    )

    documents = results.get("documents", [[]])[0]

    if not documents:
        return "I couldn't find relevant car information in my database."

    context = "\n".join(documents)

    # Build prompt
    prompt = f"""
    You are a professional and friendly car salesman.

    Instructions:
    - Use ONLY the provided context.
    - Answer the user's question accurately.
    - Be concise and helpful.
    - Do not invent car specifications.
    - If the answer is not in the context, say so.
    - Format using Markdown headings, bold text, and bullet points.

    Context:
    {context}

    User question:
    {query}
    """

    # Generate response
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# 5. Text-to-Speech Function
# -----------------------------

def generate_audio(text):

    clean_text = (
        text.replace("#", "")
            .replace("*", "")
    )

    audio_buffer = io.BytesIO()

    tts = gTTS(
        text=clean_text,
        lang="en"
    )

    tts.write_to_fp(audio_buffer)

    audio_buffer.seek(0)

    return audio_buffer


# -----------------------------
# 6. Chat History
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and "audio" in message:
            st.audio(
                message["audio"],
                format="audio/mp3"
            )


# -----------------------------
# 7. User Input
# -----------------------------

if query := st.chat_input(
    "Ask about cars... e.g. Best 4-cylinder car for a student?"
):

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Searching car database..."):

            try:
                answer = chat_with_cars(query)

                st.markdown(answer)

                # Generate speech
                with st.spinner("Generating voice..."):
                    audio = generate_audio(answer)
                    audio_bytes = audio.getvalue()

                # Audio player
                st.audio(
                    audio_bytes,
                    format="audio/mp3",
                    autoplay=True
                )

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "audio": audio_bytes
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
