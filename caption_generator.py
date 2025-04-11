import os
import openai
from dotenv import load_dotenv

load_dotenv()  # This will load the variables from the .env file

openai_api_key = os.getenv("OPENAI_API_KEY")

# Set your OpenAI API key from the environment
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_caption(image_description: str, location: str = "", tone: str = "", additional_context: str = "") -> str:
    """
    Generate a short, modern, and succinct Instagram caption using OpenAI's gpt-3.5-turbo model.
    """
    prompt = (
        f"Create a modern and engaging Instagram caption based on the details below. "
        f"Keep the caption short (about 5-8 words), with no emojis, and no hashtags\n"
        f"Image description: {image_description}\n"
    )
    if location:
        prompt += f"Location: {location}\n"
    if tone:
        prompt += f"Tone: {tone}\n"
    if additional_context:
        prompt += f"Additional context: {additional_context}\n"
    prompt += "Caption:"

    messages = [
        {"role": "system", "content": (
            "You are an expert social media content creator who specializes in generating modern, "
            "succinct Instagram captions. Your captions are always brief (5-8 words), and they do not include emojis or hashtags."
        )},
        {"role": "user", "content": prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=25,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        caption = response["choices"][0]["message"]["content"].strip()
        return caption
    except Exception as e:
        return f"Error generating caption: {e}"


def generate_alternative_prompts(final_caption: str, feedback: str, direction: str) -> list:
    """
    Generate three alternative Instagram caption prompts based on the user's final caption, feedback, and direction.
    """
    prompt = (
        "You are an expert social media content creator who specializes in generating captions. "
        "A user has provided the following final Instagram caption: "
        f"'{final_caption}'. The user feedback is: '{feedback}' and the additional direction is: '{direction}'. "
        "Based on this, please generate three new, creative Instagram caption prompts that are always brief (5-8 words), and do not include emojis or hashtags. "
        "Each prompt should be distinct and provide a new angle for the caption."
    )

    messages = [
        {"role": "system", "content": "You are a creative social media strategist."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.8,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        text = response["choices"][0]["message"]["content"].strip()
        # Assume that the response contains three separate prompts (e.g., each on its own line)
        prompts = [line.strip() for line in text.split("\n") if line.strip()]
        return prompts
    except Exception as e:
        return [f"Error generating alternative prompts: {e}"]


if __name__ == "__main__":
    # Example usage
    image_desc = "a group of young men sitting on a rock in a park"
    caption = generate_caption(image_desc)
    print("Generated Caption:", caption)
    # Example feedback generation
    feedback = "I think it lacks energy."
    direction = "Make it more vibrant and youthful."
    alt_prompts = generate_alternative_prompts(caption, feedback, direction)
    print("Alternative Prompts:")
    for prompt in alt_prompts:
        print(prompt)
