# Use the official Python 3.9 slim image as the base
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and to buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Gradio uses
EXPOSE 7860

# Define environment variable for OpenAI API Key (to be provided at runtime)
ENV OPENAI_API_KEY=""

# Specify the command to run the application
CMD ["python", "app.py"]
