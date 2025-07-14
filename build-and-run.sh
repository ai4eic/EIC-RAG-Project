#!/bin/bash

# Build and run script for EIC-RAG Streamlit app

set -e

# Check if secrets file argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path-to-secrets.toml>"
    echo ""
    echo "Example: $0 /path/to/your/secrets.toml"
    echo ""
    echo "The secrets.toml file should contain:"
    echo "OPENAI_API_KEY = \"your_openai_key\""
    echo "PINECONE_API_KEY = \"your_pinecone_key\""
    echo "ADMIN_EMAIL = \"your_admin_email\""
    echo "ADMIN_PASSWORD = \"your_admin_password\""
    exit 1
fi

SECRETS_FILE="$1"

# Check if secrets.toml exists at provided path
if [ ! -f "$SECRETS_FILE" ]; then
    echo "Error: Secrets file not found at: $SECRETS_FILE"
    echo "Please provide a valid path to your secrets.toml file."
    exit 1
fi

echo "Building EIC-RAG Streamlit Docker image..."

# Build the Docker image
docker build -f DockerFile -t eic-rag-streamlit:latest .

echo "Docker image built successfully!"
echo "Starting the container..."

# Stop and remove existing container if it exists
docker stop eic-rag-app 2>/dev/null || true
docker rm eic-rag-app 2>/dev/null || true

# Run the container
docker run -d \
    --name eic-rag-app \
    -p 8502:8502 \
    -v "$SECRETS_FILE:/app/.streamlit/secrets.toml:ro" \
    eic-rag-streamlit:latest

echo "Container started successfully!"
echo "Waiting for container to be ready..."

# Wait a moment for container to start
sleep 5

# Check container status
if ! docker ps | grep -q eic-rag-app; then
    echo "ERROR: Container is not running!"
    echo "Container logs:"
    docker logs eic-rag-app
    exit 1
fi

echo "Container is running. Checking application health..."
echo "Container logs (last 20 lines):"
docker logs --tail 20 eic-rag-app

echo ""
echo "Access the app at: http://localhost:8502"
echo ""
echo "Debugging commands:"
echo "  Check container status: docker ps"
echo "  View live logs: docker logs -f eic-rag-app"
echo "  Execute into container: docker exec -it eic-rag-app /bin/bash"
echo "  Stop container: docker stop eic-rag-app"
echo "  Remove container: docker rm eic-rag-app"
