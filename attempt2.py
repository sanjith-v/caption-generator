from fastapi import FastAPI, File, UploadFile
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch

app = FastAPI()

# Load your pretrained/fine-tuned model
# Example: GPT-2
tokenizer = AutoTokenizer.from_pretrained("path/to/your-fine-tuned-model")
model = AutoModelForCausalLM.from_pretrained("path/to/your-fine-tuned-model")

# If you have an image captioning model
# from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# image_captioning_model = VisionEncoderDecoderModel.from_pretrained("...")
# image_processor = ViTImageProcessor.from_pretrained("...")
# caption_tokenizer = AutoTokenizer.from_pretrained("...")


def generate_keywords(image: Image):
    # Some function that returns keywords or a short description from the image
    return ["beach", "sunset"]


def generate_caption(keywords):
    prompt = f"Generate an Instagram caption with references to pop culture. Keywords: {', '.join(keywords)}. Caption:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=60,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


@app.post("/generate_caption")
async def caption_endpoint(file: UploadFile = File(...)):
    # 1. Read image
    image = Image.open(file.file).convert("RGB")

    # 2. Get keywords from the image
    keywords = generate_keywords(image)

    # 3. Generate the creative caption
    caption = generate_caption(keywords)

    return {"keywords": keywords, "caption": caption}
