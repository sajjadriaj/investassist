# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Expose port 5000
EXPOSE 5000

# Define environment variables for API keys
# These will be set when running the container
ENV NEWS_API_KEY=your_news_api_key_here
ENV GEMINI_API_KEY=your_gemini_api_key_here

# Run the application
CMD ["python", "app.py"]
