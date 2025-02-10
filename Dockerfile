# Use an official Python runtime as a parent image
FROM python:3.12-slim

WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install waitress

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

CMD python app.py