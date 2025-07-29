# 🔋 Genero Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
Genero Toolkit is a multi-functional AI-powered application designed for processing and analyzing various data sources with optimized performance and minimal resource usage. It leverages lightweight transformer models to provide three core functionalities:
1. **YouTube Video Summarization**: Summarizes YouTube video transcripts using the T5-small model.
2. **PDF Question Answering**: Extracts text from PDFs and answers questions using MobileBERT.
3. **Sentiment Analysis**: Analyzes sentiments in Excel review data using BERTweet.

This project is optimized for efficiency, utilizing models with a combined footprint of approximately 300MB, and supports GPU acceleration when available.

## Features
- **YouTube Summarizer**: Extracts transcripts from YouTube videos and generates concise summaries.
- **PDF Q&A**: Upload a PDF and ask questions about its content with context-aware answers.
- **Sentiment Analysis**: Processes Excel files containing reviews, categorizes sentiments (Positive, Neutral, Negative), and visualizes results with a pie chart.
- **User Interface**: Built with Gradio for an intuitive, tab-based experience.
- **Optimized Performance**: Uses lightweight models (T5-small, MobileBERT, BERTweet) for fast inference and low memory usage.
- **Hardware Acceleration**: Automatically utilizes CUDA if a compatible GPU is available.

## Requirements
To run the Genero Toolkit, ensure you have the following dependencies installed:
```bash
pip install torch gradio pandas matplotlib youtube_transcript_api transformers PyPDF2
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MemaroX/Genero.git
   cd Genero
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

## Usage
1. **Launch the Application**:
   - Run `python main.py` to start the Gradio interface in your browser.
2. **YouTube Summarizer**:
   - Paste a YouTube URL and click "Summarize" to generate a summary of the video's transcript.
3. **PDF Q&A**:
   - Upload a PDF file, then enter a question about its content and click "Get Answer".
4. **Sentiment Analysis**:
   - Upload an Excel file with a "Reviews" column, then click "Analyze" to view sentiment results and a pie chart.

## Project Structure
```
Genero/
├── Genero-report.pdf   # Detailed project report or methodology
├── main.py             # Main application script (Gradio interface)
├── README.md           # Project documentation
├── requirements.txt    # List of Python dependencies
└── xl.py               # Utility script, likely for Excel processing
```

## Models
The toolkit uses the following pre-trained models from Hugging Face:
- **T5-small** (~60MB): For YouTube video summarization.
- **MobileBERT** (~100MB): For PDF question answering.
- **BERTweet** (~140MB): For sentiment analysis of reviews.

## Limitations
- YouTube summarization requires videos to have English transcripts available.
- PDF Q&A performance depends on the quality of text extraction from the PDF.
- Sentiment analysis assumes the Excel file has a "Reviews" column.
- Large inputs may require chunking to manage memory efficiently.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built as a final project to demonstrate AI-driven data processing.
- Powered by Hugging Face Transformers, Gradio, and YouTube Transcript API.
- Special thanks to the open-source community for providing lightweight models and tools.

## Contact

MemaroX - [Your GitHub Profile Link](https://github.com/MemaroX]