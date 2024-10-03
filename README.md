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
