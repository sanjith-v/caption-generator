# Instagram Caption Generator

## Overview

A GenAI-powered tool that automatically generates engaging Instagram captions from images. Built as part of my data science portfolio, this project showcases end-to-end expertise in computer vision, NLP, and scalable API development using Huggingface and OpenAI.

## Live Demo

Check out the live deployment here: [Instagram Caption Generator](https://caption-generator-24b124df5c0e.herokuapp.com)

## Key Features

- **Image Understanding**: Uses Huggingface’s BLIP model to extract rich visual features and generate a concise scene description.
- **Caption Refinement**: Feeds BLIP output into OpenAI’s GPT-4 Vision API (or `openai.Image.createCaption`) to produce creative, context-aware captions.
- **High Throughput API**: Exposes a RESTful endpoint (FastAPI) that accepts image uploads and returns captions in JSON—designed for low latency and horizontal scaling.
- **Containerized Deployment**: Dockerized with multi-stage builds to minimize image size; ready for deployment on Render, Heroku, or Google Cloud Run.
- **Logging & Monitoring**: Integrated with Prometheus metrics and structured logging for real-time performance tracking and error alerts.

## Tech Stack

- **Modeling & Inference**
  - 🤗 transformers (BLIP)
  - OpenAI Python SDK (GPT-4 Vision & text generation)
- **API & Backend**
  - FastAPI with Uvicorn
  - Pydantic for payload validation
- **Deployment & DevOps**
  - Docker & Docker Compose
  - CI/CD with GitHub Actions
  - Hosted on Render (or GCR/Heroku)
- **Testing & Quality**
  - PyTest for unit & integration tests
  - Black & isort for code formatting

## Getting Started

1. **Clone & Install**
   ```bash
   git clone https://github.com/yourusername/instagram-caption-generator.git
   cd instagram-caption-generator
   pip install -r requirements.txt
   ```

## Environment Variables

```bash
export OPENAI_API_KEY="your-openai-key"
```

## Run Locally

```bash
uvicorn app.main:app --reload
```

## Generate a Caption

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "image=@/path/to/photo.jpg" \
  -H "Accept: application/json"
```

## Deployment

```bash
# Build Docker image
docker build -t caption-generator:latest .

# Tag & push to your container registry
docker tag caption-generator:latest your-registry/caption-generator:latest
docker push your-registry/caption-generator:latest

# Deploy on Render / Google Cloud Run / Heroku
# Configure your service to pull the above image; it will auto-scale based on request load
```
