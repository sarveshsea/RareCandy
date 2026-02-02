# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies required for building Python packages (numpy/pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Make sure the start script is executable
RUN chmod +x start_exposed.sh

# Expose port 8000 for the telemetry dashboard
EXPOSE 8000

# Define environment variable
# ENV NAME RareCandy

# Run start_exposed.sh when the container launches
CMD ["./start_exposed.sh"]
