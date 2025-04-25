"""
image_analysis.py

Uses the Hugging Face Inference API to generate a detailed description of an image.
Maintains the same function signature and example usage as before.
"""

import os
import time
from huggingface_hub import InferenceClient
from huggingface_hub.inference._generated.types.image_to_text import ImageToTextOutput

# ------------------------------------------------------------------------------
# Global setup:
# 1. Instantiate the HF Inference client once using your HF token.
# ------------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)


def describe_image(image_path: str, prompt: str = "Describe the image in detail.") -> str:
    """
    Sends an image to HF’s Inference API and returns a clean description string.

    Args:
        image_path (str): The path to the image file.
        prompt (str): An optional prompt to guide the captioning (currently ignored by the API).

    Returns:
        str: A detailed description of the image.
    """
    # Call the HF image-to-text endpoint
    output = client.image_to_text(
        image=image_path, model="Salesforce/blip-image-captioning-large")

    # Normalize whatever type comes back into a single string
    if isinstance(output, str):
        text = output

    elif isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, str):
            text = first
        elif hasattr(first, "generated_text"):
            text = first.generated_text
        elif hasattr(first, "image_to_text_output_generated_text"):
            text = first.image_to_text_output_generated_text
        else:
            text = str(first)

    elif isinstance(output, ImageToTextOutput):
        # Prefer .generated_text, fallback to .image_to_text_output_generated_text
        text = (
            output.generated_text
            if output.generated_text is not None
            else output.image_to_text_output_generated_text or ""
        )

    else:
        text = str(output)

    return text.strip()


if __name__ == "__main__":
    # Example usage (uncomment to test locally):
    test_image = "test_pictures/WhatsApp Image 2025-04-02 at 15.26.28.jpeg"
    start = time.time()
    caption = describe_image(test_image)
    end = time.time()

    print("Caption:", caption)
    print("Time taken:", end - start)
