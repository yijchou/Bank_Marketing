# Use an official Python runtime as a parent image
FROM python:3.11.5-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# Assuming requirements.txt is in the web_app directory
RUN pip install --no-cache-dir -r web_app/requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Change the working directory to /app/web_app
WORKDIR /app/web_app

# Run the Streamlit app using the module (python -m) invocation
CMD ["python3", "-m", "streamlit", "run", "Homepage.py"]

