"""
image_analysis.py

Uses the BLIP model from Salesforce to generate a detailed description of an image.
Optimized for speed while maintaining comprehensive output.
"""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

# ------------------------------------------------------------------------------
# Global setup for speed:
# 1. Load the BLIP model and processor once.
# 2. Use GPU if available and convert model to half precision.
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large")
model.to(device)

# Convert to half precision if using GPU for faster inference.
if device.type == "cuda":
    model.half()

model.eval()


@torch.no_grad()
def describe_image(image_path: str, prompt: str = "Describe the image in detail.") -> str:
    """
    Generates a detailed description of the image using the BLIP model.

    Args:
        image_path (str): The path to the image file.
        prompt (str): An optional prompt to guide the captioning.

    Returns:
        str: A detailed description of the image.
    """
    # Open, convert to RGB, and resize the image to speed up processing.
    image = Image.open(image_path).convert('RGB')
    # Resize to a typical resolution for captioning.
    image = image.resize((384, 384))

    # Prepare inputs using the processor.
    inputs = processor(image, prompt, return_tensors="pt").to(device)

    # Generate the description using fewer beams and a shorter max_length to reduce runtime.
    output_ids = model.generate(**inputs, max_length=80, num_beams=4)
    description = processor.decode(output_ids[0], skip_special_tokens=True)
    print(description)

    return description


if __name__ == '__main__':
    # Example usage (uncomment to test locally):
    test_image = "/Users/sanjithkrishna/Desktop/Coding/portfolio/captionGenerator/test_pictures/DSC00622 copy 2.JPG"
    start = time.time()
    desc = describe_image(test_image)
    end = time.time()
    print("Description:", desc)
    print("Time taken:", end - start)
