import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_path = "/Users/sanjithkrishna/Desktop/Coding/portfolio/captionGenerator/images/c8e5a53e-1b10-4938-9d9c-40ff19b86529 copy.jpg"
image = Image.open(image_path)
processed_image = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(processed_image)

image_features /= image_features.norm(dim=-1, keepdim=True)

candidate_texts = [
    "a vibrant sunrise",                # Early morning, natural light
    "a majestic sunset",                 # Evening with warm hues
    "a dynamic cityscape",              # Urban environment with skyscrapers
    "a bustling urban street",          # Busy street scene, full of life
    "a serene countryside",             # Calm rural or nature scene
    "a scenic mountain view",           # Majestic mountains or high-altitude scenes
    "a calm beach scene",               # Relaxed seaside or coastal view
    "a lively nightclub scene",         # Nightclub with energetic, blurred lights
    "a blurred party atmosphere",       # Motion-filled, dynamic party environment
    # Action-packed moments like skydiving or surfing
    "a thrilling adventure sport moment",
    "an intense action shot",           # High-energy sports or adventure activity
    "a joyful family gathering",        # Warm, intimate family event
    "a group of close friends",         # Casual get-together with friends
    "an energetic festival",            # Vibrant public celebration or fair
    "a sophisticated office environment",  # Modern work or business setting
    "a quaint coffee shop",             # Cozy and inviting café scene
    "a modern art gallery",             # Contemporary and creative indoor space
    "a mysterious, foggy scene",        # Ambiguous, atmospheric visual with fog
    "a colorful street market",         # Vibrant market with diverse elements
    "a peaceful park setting",          # Tranquil outdoor public space
    "a dramatic natural landscape",     # Powerful, awe-inspiring nature scene
    "a romantic evening ambiance"       # Intimate, soft-lit dinner or date setting
]

text_tokens = clip.tokenize(candidate_texts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

similarities = (image_features @ text_features.T).squeeze(0)
# Convert to a list of tuples: (text, similarity_score)
ranked_texts = sorted(
    zip(candidate_texts, similarities.tolist()),
    key=lambda x: x[1],
    reverse=True
)
print("Ranked candidate descriptions:", ranked_texts)
