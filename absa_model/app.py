
import torch
import gradio as gr
from transformers import BertTokenizerFast, AutoModelForSequenceClassification

import os

MODEL_PATH = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cpu")

print("Loading ABSA model...")

# Load tokenizer and trained model
tokenizer = BertTokenizerFast(
    tokenizer_file=f"{MODEL_PATH}/tokenizer.json"
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.to(device)
model.eval()

print("Model loaded successfully!")
print("Device:", device)
print("Labels:", model.config.id2label)


def predict_sentiment(review, aspect):
    # Validate input
    if not review or not review.strip():
        return {"Error": 1.0}

    if not aspect or not aspect.strip():
        return {"Please enter an aspect": 1.0}

    # Tokenize review and aspect as a pair
    inputs = tokenizer(
        review.strip(),
        aspect.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Predict sentiment
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    # Return all four class probabilities
    return {
        model.config.id2label[i]: round(
            probabilities[i].item(), 4
        )
        for i in range(len(probabilities))
    }


# Gradio interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=[
        gr.Textbox(
            label="Customer Review",
            placeholder="The camera is excellent, but the battery is poor.",
            value="The camera is excellent, but the battery is poor."
        ),
        gr.Textbox(
            label="Aspect to Analyze",
            placeholder="Enter an aspect, e.g. camera",
            value="camera"
        )
    ],
    outputs=gr.Label(
        num_top_classes=4,
        label="Sentiment Probabilities"
    ),
    title="Aspect-Based Sentiment Analysis",
    description=(
        "Enter a customer review and a specific aspect. "
        "The trained DistilBERT model predicts positive, "
        "negative, neutral, or conflict sentiment for that aspect."
    ),
    examples=[
        [
            "The camera is excellent, but the battery is poor.",
            "camera"
        ],
        [
            "The camera is excellent, but the battery is poor.",
            "battery"
        ],
        [
            "The food was okay.",
            "food"
        ]
    ]
)


if __name__ == "__main__":
    demo.launch()
