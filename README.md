# Aspect-Based Sentiment Analysis (ABSA)

## 📌 Project Overview
This project implements an Aspect-Based Sentiment Analysis (ABSA) system using Natural Language Processing (NLP) and Transformer-based models. It analyzes text to identify specific aspects and determine the sentiment expressed toward each aspect.

## 🎯 Objectives
- Identify important aspects mentioned in user reviews or text.
- Classify sentiment associated with individual aspects.
- Provide an interactive interface for users to analyze text.
- Demonstrate the practical application of NLP and deep learning.

## ✨ Features
- Text input for sentiment analysis.
- Aspect-level sentiment analysis.
- Transformer-based NLP model.
- Interactive Gradio user interface.
- Configured tokenizer and model settings.

## 🛠️ Technologies Used
- Python
- Natural Language Processing (NLP)
- Transformer Models
- Gradio
- PyTorch / Hugging Face Transformers *(include only those actually used in your code)*

## 📂 Project Structure
```text
ABSA-Project/
├── absa_model/
│   ├── app.py
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── training_args.bin
│   └── model.safetensors  # Stored separately; not included in GitHub
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/kusumavangala/ABSA-Project.git
   cd ABSA-Project
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:
   ```bash
   pip install gradio torch transformers
   ```

4. Download or place the required trained model files in the appropriate location.

5. Run the application:
   ```bash
   python absa_model/app.py
   ```

## 🚀 Applications
- Analyzing customer reviews.
- Understanding product-specific feedback.
- Extracting aspect-level opinions from text.
- Supporting customer feedback analysis.

## 🔮 Future Enhancements
- Deploy the application online.
- Improve model performance using additional datasets.
- Add visualizations for aspect-level sentiment results.
- Support batch analysis of multiple reviews.

## 👨‍💻 Author
**Kusumavangala**

GitHub: https://github.com/kusumavangala

## 📄 Note
The trained model weights are not included in this repository. The application requires the compatible model files to run successfully.
