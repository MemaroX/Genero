import os
import re
import torch
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from PyPDF2 import PdfReader

# ===== Configuration =====
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN warnings

# Optimized models configuration
MODELS = {
    "summarization": ("t5-small", 60),
    "sentiment": ("finiteautomata/bertweet-base-sentiment-analysis", 140),
    "qa": ("mrm8488/mobilebert-uncased-finetuned-squadv2", 100)
}

# ===== Model Loading =====
def load_models():
    """Load all models with minimal memory footprint"""
    print("⚡ Loading optimized models...")
    models = {}
    
    try:
        # Summarization model
        models["summarization"] = pipeline(
            "text2text-generation",
            model=MODELS["summarization"][0],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Sentiment analysis
        models["sentiment"] = pipeline(
            "text-classification",
            model=MODELS["sentiment"][0],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Question Answering
        models["qa"] = pipeline(
            "question-answering",
            model=MODELS["qa"][0],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"✅ Models loaded (Total ~{sum(m[1] for m in MODELS.values())}MB)")
        return models
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        raise

models = load_models()

# ===== YouTube Functions =====
def extract_video_id(url):
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url or "")
        if match:
            return match.group(1)
    return None

def summarize_youtube(url):
    video_id = extract_video_id(url)
    if not video_id:
        return None, "❌ Invalid YouTube URL"
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        text = " ".join([t['text'] for t in transcript])
        
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        summaries = []
        
        for chunk in chunks:
            result = models["summarization"](
                "summarize: " + chunk,
                max_length=130,
                min_length=30,
                do_sample=False
            )
            summaries.append(result[0]['generated_text'])
        
        return " ".join(summaries), None
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ===== PDF Functions =====
pdf_context = ""

def process_pdf(file, question):
    global pdf_context
    if file:
        try:
            reader = PdfReader(file.name)
            pdf_context = "\n".join([page.extract_text() for page in reader.pages])
            return f"📄 Loaded {len(pdf_context)} characters", ""
        except Exception as e:
            return f"❌ PDF Error: {str(e)}", ""
    
    if question and pdf_context:
        try:
            answer = models["qa"](question=question, context=pdf_context)
            return "", answer["answer"]
        except Exception as e:
            return "", f"❌ QA Error: {str(e)}"
    return "", "❌ Upload PDF first"

# ===== Sentiment Analysis =====
def analyze_reviews(file):
    try:
        df = pd.read_excel(file.name)
        if 'Reviews' not in df.columns:
            return None, None, "❌ File must contain 'Reviews' column"
        
        chunk_size = 32
        chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        results = []
        
        for chunk in chunks:
            predictions = models["sentiment"](chunk['Reviews'].astype(str).tolist())
            chunk['Sentiment'] = [
                "POSITIVE" if pred['label'] == "POS" else 
                "NEGATIVE" if pred['label'] == "NEG" else 
                "NEUTRAL"
                for pred in predictions
            ]
            results.append(chunk)
        
        df = pd.concat(results)
        
        sentiment_order = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
        sentiment_counts = df['Sentiment'].value_counts().reindex(sentiment_order, fill_value=0)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#4CAF50', '#FFC107', '#F44336']
        wedges, texts, autotexts = ax.pie(
            sentiment_counts,
            autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
            colors=colors,
            startangle=90,
            pctdistance=0.85
        )
        
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax.add_artist(centre_circle)
        
        ax.legend(
            wedges,
            [f"{label} ({count})" for label, count in zip(sentiment_counts.index, sentiment_counts)],
            title="Sentiment",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        ax.set_title('Sentiment Distribution', pad=20)
        plt.tight_layout()
        plt.close()
        
        return df, fig, None
        
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

# ===== Gradio Interface =====
with gr.Blocks(title="Genero Toolkit", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧠 Multi-Function AI Toolkit")
    gr.Markdown("Optimized for performance and accuracy As a final project")
    
    with gr.Tabs():
        # YouTube Summarization Tab
        with gr.Tab("🎬 YouTube Summarizer"):
            gr.Markdown("Summarize videos using T5-small (60MB)")
            with gr.Row():
                yt_url = gr.Textbox(label="YouTube URL", placeholder="Paste any YouTube link...")
                yt_btn = gr.Button("Summarize", variant="primary")
            yt_output = gr.Textbox(label="Summary", lines=6)
            yt_status = gr.Textbox(label="Status", interactive=False)
            
            yt_btn.click(
                lambda url: summarize_youtube(url),
                inputs=yt_url,
                outputs=[yt_output, yt_status]
            )
        
        # PDF Q&A Tab
        with gr.Tab("📄 PDF Q&A"):
            gr.Markdown("Ask questions about PDFs using MobileBERT (100MB)")
            with gr.Row():
                pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
            with gr.Row():
                pdf_question = gr.Textbox(label="Your Question", placeholder="Ask about the PDF content...")
                pdf_btn = gr.Button("Get Answer", variant="primary")
            pdf_answer = gr.Textbox(label="Answer")
            pdf_status = gr.Textbox(label="Status", interactive=False)
            
            pdf_file.change(
                lambda x: process_pdf(x, ""),
                inputs=pdf_file,
                outputs=[pdf_status, pdf_answer]
            )
            pdf_btn.click(
                lambda q: process_pdf(None, q),
                inputs=pdf_question,
                outputs=[pdf_status, pdf_answer]
            )
        
        # Sentiment Analysis Tab
        with gr.Tab("😊 Sentiment Analysis"):
            gr.Markdown("Analyze review sentiments using BERTweet (140MB)")
            with gr.Row():
                review_file = gr.File(label="Upload Excel", file_types=[".xlsx"])
                review_btn = gr.Button("Analyze", variant="primary")
            with gr.Row():
                review_table = gr.Dataframe(label="Results")  # Removed height parameter
                review_chart = gr.Plot()
            review_status = gr.Textbox(label="Status", interactive=False)
            
            review_btn.click(
                analyze_reviews,
                inputs=review_file,
                outputs=[review_table, review_chart, review_status]
            )

if __name__ == "__main__":
    app.launch()