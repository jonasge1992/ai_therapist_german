# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies including FFmpeg
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Make port 4221 available to the world outside this container
EXPOSE 4221

# Copy the rest of the application's code into the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the NLTK data download script into the container
COPY download_nltk_data.py .

# Run the NLTK data download script
RUN python download_nltk_data.py

# Run the application
CMD ["python", "main.py"]
