import gradio as gr
from transformers import pipeline
import evaluate

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def summarize_text(input_text):
    result = summarizer(
        input_text,
        max_length=45,
        min_length=15,
        do_sample=False
    )
    return result[0]["summary_text"]

demo = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=8, label="Enter text to summarize"),
    outputs=gr.Textbox(label="Generated Summary"),
    title="GenAI Text Summarizer",
    description="A cloud-deployable Generative AI summarization app built with Gradio."
)

# Set share=True to generate a public Gradio URL.
demo.launch(share=True)

rouge = evaluate.load("rouge")

generated_summaries = [
    "AI models generate new content such as text and images."
]

reference_summaries = [
    "Generative AI models are capable of producing new content including text and images."
]

scores = rouge.compute(
    predictions=generated_summaries,
    references=reference_summaries
)

print("ROUGE Evaluation Scores:", scores)
