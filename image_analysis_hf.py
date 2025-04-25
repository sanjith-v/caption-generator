import os
from huggingface_hub import InferenceClient
from huggingface_hub.inference._generated.types.image_to_text import ImageToTextOutput

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)


def describe_image_hf(image_path: str, model: str = "Salesforce/blip-image-captioning-large") -> str:
    """
    Sends an image to HF’s Inference API and returns a clean caption string,
    no matter what Python type the client returns under the hood.
    """
    output = client.image_to_text(image=image_path, model=model)

    # 1) If it’s already a str, use it directly
    if isinstance(output, str):
        text = output

    # 2) If it’s a list (of strings or objects), grab the first element
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

    # 3) If it’s the typed ImageToTextOutput object
    elif isinstance(output, ImageToTextOutput):
        # that class defines both .generated_text and .image_to_text_output_generated_text
        text = (
            output.generated_text
            if output.generated_text is not None
            else output.image_to_text_output_generated_text
        )

    # 4) Fallback for anything else
    else:
        text = str(output)

    return text.strip()


if __name__ == "__main__":
    img_path = "test_pictures/WhatsApp Image 2025-04-02 at 15.26.28.jpeg"
    caption = describe_image_hf(img_path)
    print("Caption:", caption)
