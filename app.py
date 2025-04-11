"""
app.py

A FastAPI app that:
1) Accepts an image upload.
2) Describes the image via BLIP in the background.
3) Allows the user to provide additional context.
4) Generates a final caption.
5) Records user feedback and generates three alternative caption prompts.
"""

import os
import uuid
import time
from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from image_analysis import describe_image
from caption_generator import generate_caption, generate_alternative_prompts
import uvicorn

app = FastAPI()

# Mount static files (CSS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up the templates directory
templates = Jinja2Templates(directory="templates")

# Global dictionary to store image analysis results keyed by uid
analysis_results = {}


def process_image(uid: str, filename: str):
    """
    Performs image analysis in the background and stores the result.
    """
    result = describe_image(filename)
    analysis_results[uid] = result
    # Clean up the temporary file
    if os.path.exists(filename):
        os.remove(filename)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """
    Serves the initial image upload page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_image")
async def upload_image(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Saves the uploaded image, starts the background image analysis, and
    returns a processing page that will redirect to the context page after 5 seconds.
    """
    uid = str(uuid.uuid4())
    temp_filename = f"temp_{uid}.jpg"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    # Launch image analysis in the background.
    background_tasks.add_task(process_image, uid, temp_filename)

    # Render a processing page that waits 5 seconds then redirects to /context.
    return templates.TemplateResponse("processing.html", {"request": request, "uid": uid})


@app.get("/context", response_class=HTMLResponse)
def get_context(request: Request, uid: str):
    """
    Renders the additional context page.
    The raw description is not displayed; only the uid is passed along.
    """
    return templates.TemplateResponse("context.html", {"request": request, "uid": uid})


@app.post("/generate_caption", response_class=HTMLResponse)
def generate_caption_route(
    request: Request,
    uid: str = Form(...),
    location: str = Form(""),
    tone: str = Form(""),
    additional_context: str = Form("")
):
    """
    Retrieves the image analysis result using the uid (waiting briefly if needed),
    then generates and displays the final caption without showing the raw description.
    """
    # Wait up to 10 seconds for the image analysis result if it is not ready.
    wait_time = 0
    while uid not in analysis_results and wait_time < 10:
        time.sleep(1)
        wait_time += 1

    raw_description = analysis_results.get(uid, "No description available.")
    # Optionally, remove the entry after usage.
    analysis_results.pop(uid, None)

    final_caption = generate_caption(
        image_description=raw_description,
        location=location,
        tone=tone,
        additional_context=additional_context
    )
    return templates.TemplateResponse("final.html", {"request": request, "caption": final_caption})


@app.post("/feedback", response_class=HTMLResponse)
def feedback_route(
    request: Request,
    final_caption: str = Form(...),
    feedback: str = Form(...),
    direction: str = Form("")
):
    """
    Records user feedback and generates three alternative caption prompts.
    """
    alt_prompts = generate_alternative_prompts(
        final_caption, feedback, direction)
    return templates.TemplateResponse("feedback_result.html", {"request": request, "alt_prompts": alt_prompts})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
