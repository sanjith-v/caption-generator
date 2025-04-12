# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install build dependencies (if needed) then install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose port 8000 (Cloud Run sets the port through $PORT, but it must be exposed)
EXPOSE 8000

# Start the application using uvicorn. Note that Cloud Run will set the PORT env variable.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
