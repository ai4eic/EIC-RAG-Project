# Retrieval Augmented Generation for EIC

This is a project that is currently being developed to build a RAG based system for the upcoming EIC.  Refer to the [Description](https://github.com/ai4eic/EIC-RAG-Project/discussions/6) about the project.

## Installation

1. Create a virtual environment:

```bash
export RAG4EIC_PROJECT=/path/to/your/project
python -m venv $RAG4EIC_PROJECT/env_RAG4EIC-V0 
source $RAG4EIC_PROJECT/env_RAG4EIC-V0/bin/activate
```

2. Install Poetry:

Poetry is a dependency management tool for Python. You can install it using the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install the project dependencies:

```bash
poetry install
```

4. Clone from the repository:

```bash
git clone https://github.com/ai4eic/EIC-RAG-Project.git $RAG4EIC_PROJECT/EIC-RAG-Project
cd $RAG4EIC_PROJECT/EIC-RAG-Project
```


5. Running the webapp

    * Ask `karthik18495@gmail.com` about the `secrets.toml` and `config.toml`
    * Create a folder named `.streamlit` in the parent directory and move the files `secrets.toml` and `config.toml` in there. 
    * Now run `streamlit run streamlit_app/AI4EIC-RAGAS4EIC.py`. This should run on a `http://localhost:8050` 

## Updating 

1. If any new library has been used in the app that requires installation through pip. Make sure to use the `--format freeze` when updating the `requirements.txt`
2. The command is `pip list --format freeze > requirements.txt`


## Docker Build and Run

This project includes Docker support for easy deployment and testing.

### Prerequisites
- Docker installed on your system
- A `secrets.toml` file with your API keys and credentials

### Production Build and Run

1. Make the build script executable:
```bash
chmod +x build-and-run.sh
```

2. Build and run the application:
```bash
./build-and-run.sh /path/to/your/secrets.toml
```

3. Access the application:
- Production app runs on port **8502**
- Access at: http://localhost:8502

4. Management commands:
```bash
# Stop the container
docker stop eic-rag-app

# Remove the container
docker rm eic-rag-app

# View logs
docker logs -f eic-rag-app
```

### Docker Compose (Alternative)

You can also use Docker Compose with the helper script:

1. Make the Docker Compose script executable:
```bash
chmod +x docker-compose-run.sh
```

2. Run with your secrets file:
```bash
./docker-compose-run.sh /path/to/your/secrets.toml
```

3. Access at: http://localhost:8502

**Alternative methods:**

Using environment variable directly:
```bash
SECRETS_FILE=/path/to/your/secrets.toml docker-compose up -d
```

Using a `.env` file (create in project root):
```env
SECRETS_FILE=/path/to/your/secrets.toml
```
Then run: `docker-compose up -d`

**Management commands:**
```bash
# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart services
docker-compose restart
```

### Secrets File Format

Your `secrets.toml` file should contain:
```toml
OPENAI_API_KEY = "your_openai_key"
PINECONE_API_KEY = "your_pinecone_key" 
ADMIN_EMAIL = "your_admin_email"
ADMIN_PASSWORD = "your_admin_password"
```

### Notes
- All Docker configurations use port 8502
- Secrets file is mounted as read-only volume
- Containers automatically restart unless stopped manually
