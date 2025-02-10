# Use an official Python runtime as a parent image
FROM python:3.12-slim

WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Define environment variables (these can be overridden at runtime)
ENV PORT=5000

# Run app.py when the container launches
CMD exec gunicorn --bind :${PORT} app:app