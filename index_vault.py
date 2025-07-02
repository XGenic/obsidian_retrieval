
import os
import chromadb
from openai import OpenAI
import time

# --- CONFIGURATION ---
# 1. Load API Key from environment variables.
#    Create a .env file in this directory and add your key:
#    OPENAI_API_KEY="sk-YourActualKey..."
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

# 2. Set the absolute path to your Obsidian vault.
OBSIDIAN_VAULT_PATH = "C:/Users/Denis/Documents/Obsidian Vault"

# 3. Set the path where you want to store your ChromaDB database.
CHROMA_DB_PATH = "D:/Documents/chromadb"

# 4. (Optional) Name for the collection within ChromaDB.
COLLECTION_NAME = "obsidian_vault_main"


# --- CHROMA DB EMBEDDING FUNCTION ---
# This class handles the interaction with the OpenAI Embedding API.
class OpenAIEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key, model="text-embedding-3-small"):
        if not api_key:
            raise ValueError("OpenAI API key is missing. Please set it in the configuration.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def __call__(self, input_texts: list[str]) -> list[list[float]]:
        """
        This is called by ChromaDB to get embeddings.
        It sends the text to OpenAI and returns the numerical vectors.
        """
        # Replace empty strings with a single space, as the API rejects empty strings.
        sanitized_input = [text if text.strip() else " " for text in input_texts]
        
        try:
            response = self.client.embeddings.create(
                input=sanitized_input,
                model=self.model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error calling OpenAI Embedding API: {e}")
            # Return a list of empty lists with the correct dimensions to avoid crashing.
            return [[] for _ in input_texts]


# --- MAIN SCRIPT LOGIC ---
def main():
    """
    Main function to run the vault indexing process.
    """
    print("--- Starting Obsidian Vault Indexing ---")

    # 1. Initialize the embedding function with your API key.
    embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)

    # 2. Set up the ChromaDB client and collection.
    client_chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client_chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    # 3. Walk through every file and folder in the vault.
    for root, _, files in os.walk(OBSIDIAN_VAULT_PATH):
        for filename in files:
            if not filename.endswith(".md"):
                continue  # Skip non-markdown files

            file_path = os.path.join(root, filename)
            
            try:
                current_mtime = os.path.getmtime(file_path)
            except FileNotFoundError:
                print(f"Warning: File not found, skipping: {file_path}")
                continue

            # Check if the file needs to be indexed or updated.
            existing_record = collection.get(ids=[file_path], include=["metadatas"])
            
            is_new = not existing_record['ids']
            is_modified = not is_new and existing_record['metadatas'][0].get('mtime', 0) < current_mtime

            if is_new or is_modified:
                status = "NEW" if is_new else "MODIFIED"
                print(f"{status}: Processing '{filename}'...")

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                document_to_embed = ""
                metadata = {}

                # Apply the hybrid strategy based on the file's path.
                if "Conversations" in file_path:
                    try:
                        # Path A: For conversation files, use only the summary.
                        summary_marker = "## Overall Summary"
                        transcript_marker = "# Transcript"
                        
                        summary_start_index = content.find(summary_marker) + len(summary_marker)
                        summary_end_index = content.find(transcript_marker, summary_start_index)
                        
                        if summary_start_index > len(summary_marker) -1:
                            document_to_embed = content[summary_start_index:summary_end_index].strip()
                            metadata = {"source_type": "conversation_summary", "mtime": current_mtime, "full_path": file_path}
                        else:
                            print(f"  -> Warning: Summary marker not found in '{filename}'. Skipping.")
                            continue
                    except Exception as e:
                        print(f"  -> Error parsing summary for '{filename}': {e}. Skipping.")
                        continue
                else:
                    # Path B: For all other notes, use the full content.
                    document_to_embed = content
                    metadata = {"source_type": "full_note", "mtime": current_mtime, "full_path": file_path}

                # Upsert the document into ChromaDB. This will add it if it's new
                # or overwrite it if it's modified.
                if document_to_embed:
                    collection.upsert(
                        documents=[document_to_embed],
                        ids=[file_path],
                        metadatas=[metadata]
                    )
                    print(f"  -> Successfully indexed '{filename}'.")
                else:
                    print(f"  -> Warning: No content to embed for '{filename}'. Skipping.")

    print("--- Indexing Complete ---")


if __name__ == "__main__":
    main()
