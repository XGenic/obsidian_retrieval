
# Obsidian Vault RAG Chat

This project transforms your personal Obsidian vault into an intelligent, searchable knowledge base that you can chat with. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide context from your notes to a large language model (Grok), enabling a personalized and context-aware conversation.

## Features

- **Whole-Vault Indexing**: Scans your entire Obsidian vault, not just a single folder.
- **Hybrid Content Strategy**: Indexes the full text of short notes and only the summaries of extremely long files (like conversation transcripts) to maintain high signal-to-noise ratio.
- **Incremental Updates**: Intelligently checks for new or modified notes and only indexes what's necessary, saving time and API costs.
- **API-Powered Embeddings**: Offloads the heavy lifting of text embedding to the OpenAI API, removing the need for a powerful local GPU.
- **Tkinter GUI**: A simple and clean graphical user interface for chatting with your vault.

## How It Works

1.  **`index_vault.py`**: This script scans your Obsidian vault. It converts your notes into numerical representations (embeddings) using the OpenAI `text-embedding-3-small` model and stores them in a local ChromaDB vector database. It's smart enough to only update notes that are new or have been changed since the last run.

2.  **`grok with UI.py`**: This is the chat client. When you send a message, it first embeds your query and searches the ChromaDB database to find the most relevant notes from your vault. It then bundles this context with your query into a prompt and sends it to the xAI (Grok) API to generate a response.

## Setup and Installation

Follow these steps to get the project running.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Install Dependencies

Ensure you have Python 3 installed. Then, install the required libraries using pip:

```bash
pip install -r requirements.txt
```

*(Note: A `requirements.txt` file will be created for you, but if not, you can install the packages manually: `pip install chromadb openai python-dotenv`)*

### 3. Configure API Keys

This project requires API keys from OpenAI (for embeddings) and xAI (for chat).

1.  Create a file named `.env` in the project root directory.
2.  Copy the contents of `.env.example` into your new `.env` file.
3.  Replace the placeholder text with your actual API keys:

    ```
    OPENAI_API_KEY="sk-YourActualOpenAIKey..."
    XAI_API_KEY="xai-YourActualXaiKey..."
    ```

### 4. Configure Paths

Open both `index_vault.py` and `grok with UI.py` and adjust the following paths at the top of each file to match your system:

- `OBSIDIAN_VAULT_PATH`: The absolute path to your Obsidian vault.
- `CHROMA_DB_PATH`: The path where you want to store the vector database.

## Usage

1.  **Run the Indexer**: The first time you set up the project, you must index your entire vault. Run the indexer script from your terminal:

    ```bash
    python index_vault.py
    ```

    This may take some time depending on the size of your vault. Subsequent runs will be much faster.

2.  **Run the Chat UI**: Once indexing is complete, you can start the chat application:

    ```bash
    python "grok with UI.py"
    ```

    The GUI will appear, and you can start chatting with your notes!

--- 
