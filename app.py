import gradio as gr
import openai
import os
from rag_engine import RAGPipeline
from document_loader import extract_text

# Load your OpenAI API key securely (set this in Hugging Face Secrets)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the RAG engine
rag = RAGPipeline(api_key=openai.api_key)

# Function to process uploaded document
def process_document(file_path):
    if file_path is None:
        return None, "No file provided."
    text = extract_text(file_path)
    rag.process(text)
    return rag, text

# Function to answer a user question
def answer_qa(question, rag_state):
    if rag_state is None:
        return "Please process a document first."
    return rag_state.answer(question)

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📄 RAG Document Q&A (ChatGPT + FAISS)")

    with gr.Tab("📂 Upload Document"):
        file_input = gr.File(label="Upload a PDF or Word document", type="filepath")
        process_button = gr.Button("Process Document")
        state_output = gr.State()
        extracted_text_box = gr.Textbox(label="Extracted Text", lines=10, interactive=False)

        process_button.click(
            fn=process_document,
            inputs=file_input,
            outputs=[state_output, extracted_text_box]
        )

    with gr.Tab("🤖 Ask a Question"):
        question_input = gr.Textbox(label="Enter your question")
        ask_button = gr.Button("Get Answer")
        answer_output = gr.Textbox(label="Answer", lines=5, interactive=False)

        ask_button.click(
            fn=answer_qa,
            inputs=[question_input, state_output],
            outputs=answer_output
        )

demo.launch()
