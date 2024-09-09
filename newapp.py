import base64
import datetime
import json
import logging
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import openai
import pymupdf
import pyperclip
import streamlit as st
from PIL import Image

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Initialize OpenAI using API key from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

# OpenAI invocation function
def invoke_model(messages, max_tokens, temperature, top_p, top_k):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=1
        )
        # Correctly access the message content from the response
        return response['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error invoking model: {str(e)}")
        st.error(f"An error occurred during model invocation: {str(e)}")
        return None

# Function to compose message for OpenAI
def compose_message(user_prompt, file_paths):
    message = {"role": "user", "content": user_prompt}
    return [message]

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with NamedTemporaryFile(suffix=".pdf") as temp:
        temp.write(uploaded_file.getvalue())
        temp.seek(0)
        doc = pymupdf.open(temp.name)
        extract_text = "".join(page.get_text() for page in doc)
        return extract_text

# Function to extract text from text files
def extract_text_from_text(uploaded_file):
    extract_text = StringIO(uploaded_file.getvalue().decode("utf-8"))
    return extract_text.getvalue()

# Function to save images
def save_image(uploaded_file, file_paths):
    if uploaded_file.size > 5 * 1024 * 1024:
        logger.error("File size exceeds 5MB limit")
        st.error("File size exceeds 5MB limit")
        return
    image = Image.open(uploaded_file)
    file_path = Path("_temp_images") / uploaded_file.name
    file_path.parent.mkdir(exist_ok=True)
    image.save(file_path)
    file_paths.append(
        {
            "file_path": str(file_path),
            "file_type": uploaded_file.type,
        }
    )
    logger.info("Image saved: %s (%s)", file_path, uploaded_file.type)

def main():
    st.set_page_config(page_title="Multimodal Analysis", page_icon="analysis.png")

    # Custom CSS
    custom_css = """
    <style>
    /* Custom styling */
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    session_vars = {
        "system_prompt": None,
        "user_prompt": None,
        "analysis_time": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "max_tokens": 1000,
        "temperature": 1.0,
        "top_p": 0.999,
        "top_k": 250,
        "media_type": None,
    }

    for var, value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = value

    st.markdown("## Generative AI-powered Multimodal Analysis")

    with st.form("ad_analyze_form", clear_on_submit=False):
        st.markdown(
            "Describe the role you want the model to play, the task you wish to perform, and upload the content to be analyzed."
        )

        system_prompt_default = """You are an experienced Creative Director at a top-tier advertising agency."""
        user_prompt_default = """Analyze these four print advertisements for Mercedes-Benz sedans..."""

        st.session_state.system_prompt = st.text_area(
            "System Prompt:", value=system_prompt_default, height=50
        )
        st.session_state.user_prompt = st.text_area(
            "User Prompt:", value=user_prompt_default, height=250
        )

        uploaded_files = st.file_uploader(
            "Upload JPG, PNG, GIF, WEBP, PDF, CSV, or TXT files:",
            type=["jpg", "png", "webp", "pdf", "gif", "csv", "txt"],
            accept_multiple_files=True,
        )

        file_paths = []
        extract_text = None

        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.session_state.media_type = uploaded_file.type
                if uploaded_file.type in ["text/csv", "text/plain"]:
                    extract_text = extract_text_from_text(uploaded_file)
                    st.session_state.user_prompt += f"\n\n{extract_text}"
                elif uploaded_file.type == "application/pdf":
                    extract_text = extract_text_from_pdf(uploaded_file)
                    st.session_state.user_prompt += f"\n\n{extract_text}"
                elif uploaded_file.type in [
                    "image/jpeg",
                    "image/png",
                    "image/webp",
                    "image/gif",
                ]:
                    save_image(uploaded_file, file_paths)
                else:
                    st.error("Invalid file type. Please upload a valid file type.")

            logger.info("Prompt: %s", st.session_state.user_prompt)

        submitted = st.form_submit_button("Submit")

    if submitted and st.session_state.user_prompt:
        st.markdown("---")
        if uploaded_files:
            if uploaded_files[0].type in ["text/csv", "text/plain", "application/pdf"]:
                st.markdown(f"Sample of file contents:\n\n{extract_text[0:500]}...")
            else:
                for file_path in file_paths:
                    st.image(file_path["file_path"], caption="", width=400)

        with st.spinner(text="Analyzing..."):
            start_time = datetime.datetime.now()
            messages = compose_message(st.session_state.user_prompt, file_paths)
            if messages:
                response = invoke_model(
                    messages,
                    st.session_state.max_tokens,
                    st.session_state.temperature,
                    st.session_state.top_p,
                    st.session_state.top_k,
                )
                end_time = datetime.datetime.now()
                if response:
                    analysis = st.text_area(
                        "Model Response:", value=response, height=800
                    )
                    st.session_state.analysis_time = (
                        end_time - start_time
                    ).total_seconds()
                    pyperclip.copy(analysis)
                    st.success("Analysis copied to clipboard!")
                else:
                    st.error("An error occurred during the analysis")

    st.markdown(
        "<small style='color: #888888'> Gary A. Stafford, 2024</small>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Inference Parameters")
        st.session_state.max_tokens = st.slider(
            "max_tokens", min_value=0, max_value=5000, value=2000, step=10
        )
        st.session_state.temperature = st.slider(
            "temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05
        )
        st.session_state.top_p = st.slider(
            "top_p", min_value=0.0, max_value=1.0, value=0.999, step=0.01
        )

        st.markdown("---")

        st.text(
            f"""Inference Summary:
• max_tokens: {st.session_state.max_tokens}
• temperature: {st.session_state.temperature}
• top_p: {st.session_state.top_p}
⎯
• uploaded_media_type: {st.session_state.media_type}
⎯
• analysis_time_sec: {st.session_state.analysis_time}
"""
        )


if __name__ == "__main__":
    main()
