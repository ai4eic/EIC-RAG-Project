#!/bin/bash

# Docker Compose run script for EIC-RAG Streamlit app

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

# Convert to absolute path
SECRETS_FILE=$(realpath "$SECRETS_FILE")

echo "Using secrets file: $SECRETS_FILE"
echo "Starting EIC-RAG Streamlit app with Docker Compose..."

# Export the secrets file path and run docker-compose
export SECRETS_FILE="$SECRETS_FILE"
docker-compose up -d

echo "Container started successfully!"
echo "Access the app at: http://localhost:8502"
echo ""
echo "Management commands:"
echo "  Stop: docker-compose down"
echo "  View logs: docker-compose logs -f"
echo "  Restart: docker-compose restart"
