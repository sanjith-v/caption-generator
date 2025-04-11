from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import time

# Load the processor and BLIP model for image captioning
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large")


def describe_image(image_path, prompt="Describe the image in detail."):
    """
    Generates a detailed description of the image using the BLIP model.

    Args:
        image_path (str): The path to the image file.
        prompt (str): An optional prompt to guide the captioning.

    Returns:
        str: A detailed description of the image.
    """
    # Open and convert the image to RGB format
    image = Image.open(image_path).convert('RGB')

    # Prepare the inputs by passing both the image and the prompt
    inputs = processor(image, prompt, return_tensors="pt")

    # Generate the caption (you can adjust max_length and num_beams for more detailed output)
    output_ids = model.generate(**inputs, max_length=100, num_beams=5)

    # Decode the generated tokens to a string
    description = processor.decode(output_ids[0], skip_special_tokens=True)

    return description


# Example usage:
if __name__ == '__main__':
    # Replace with the actual image path
    image_path = "/Users/sanjithkrishna/Desktop/Coding/portfolio/captionGenerator/images/DSC00622 copy 2.JPG"
    start = time.time()
    description = describe_image(image_path)
    end = time.time()
    print("Image description:", description)
    print(end-start)
