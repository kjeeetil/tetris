# Use the official lightweight Python image.
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the application files.
COPY . /app

# Expose the port that the app runs on.
EXPOSE 8080

# Run a simple HTTP server to host the static site.
CMD ["python", "-m", "http.server", "8080"]
