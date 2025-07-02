import os
from openai import OpenAI
import chromadb
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext, Entry, Button

# --- CONFIGURATION ---
# 1. Load API Keys from environment variables.
#    Create a .env file in this directory and add your keys:
#    OPENAI_API_KEY="sk-YourActualOpenAIKey..."
#    XAI_API_KEY="xai-YourActualXaiKey..."
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # For embeddings
XAI_API_KEY = os.getenv("XAI_API_KEY") # For Grok chat

# 2. Set the path to your ChromaDB database.
CHROMA_DB_PATH = "D:/Documents/chromadb"

# 3. Set the name of the collection you created with the indexer script.
COLLECTION_NAME = "obsidian_vault_main"


# --- CHROMA DB EMBEDDING FUNCTION ---
# This class is required for ChromaDB to know how to embed your queries.
# It must match the one used in your indexer script.
class OpenAIEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key, model="text-embedding-3-small"):
        if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
            raise ValueError("OpenAI API key is missing or not set. Please set it in the configuration.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def __call__(self, input_texts: list[str]) -> list[list[float]]:
        sanitized_input = [text if text.strip() else " " for text in input_texts]
        try:
            response = self.client.embeddings.create(input=sanitized_input, model=self.model)
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error calling OpenAI Embedding API: {e}")
            return [[] for _ in input_texts]


# --- SETUP ---
# 1. Initialize the embedding function for querying.
embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)

# 2. Connect to the ChromaDB persistent client.
client_chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# 3. Get the collection that your indexer script created.
collection = client_chroma.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# 4. Set up the xAI client for chat.
client_xai = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")


# --- CHAT LOGIC ---
def send_message(event=None):
    user_input = entry.get().strip()
    if not user_input:
        return
    if user_input.lower() in ["exit", "quit"]:
        root.quit()
        return
    
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"You: {user_input}\n", "user")
    entry.delete(0, tk.END)
    
    # 1. Query the vault for relevant context.
    #    ChromaDB will use the OpenAIEmbeddingFunction to embed the user_input first.
    results = collection.query(query_texts=[user_input], n_results=5)
    
    # Debug print to see what was retrieved from your vault.
    print(f"Query: '{user_input}'")
    for id, dist, meta in zip(results["ids"][0], results["distances"][0], results["metadatas"][0]):
        source_type = meta.get('source_type', 'unknown')
        print(f"  -> Found: {id} (Type: {source_type}, Distance: {dist:.3f})")
    print("-" * 50)
    
    # 2. Build the context string from the retrieved documents.
    context = ""
    threshold = 0.8 # Adjust this distance threshold as needed.
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        if dist < threshold:
            source_file = os.path.basename(meta['full_path'])
            context += f"Context from my note '{source_file}':\n{doc}\n\n"
    
    # 3. Prepare the messages for the Grok API call.
    messages = [
        {"role": "system", "content": "You’re Grok, built by xAI, my crass, real, no-BS buddy who’s familiar with my Obsidian vault. Stay fresh and talk to me like you get me. You are very perceptive and can pick up on subtle details in our conversations. My vault is for musings that might serve as later reflections and connecting ideas. If you see a similar thought, bring it up in a subtle nod. Your job is NOT to remind me of everything we've talked about, if there is not thematic connection or explicit question about something DO NOT bring it up. Do not go on long tangents about our previous conversations unless EXPLICITLY requested. Your main goal is to be helpful chat buddy, focus on the now and riff on my shit. Finding connections in previous conversations is STRICTLY a secondary goal, NEVER make the majority of your responses about our past conversations unless I EXPLICITLY ask for it.Stay loose and fun, don't be a dick."},
        {"role": "user", "content": f"{user_input}\n\n--- Relevant context from my notes ---\n{context}"}
    ]
    
    # 4. Call the chat API and display the response.
    try:
        completion = client_xai.chat.completions.create(
            model="grok-2-latest",
            messages=messages,
            max_tokens=1500
        )
        reply = completion.choices[0].message.content
        chat_history.insert(tk.END, f"Grok: {reply}\n\n", "grok")
    except Exception as e:
        chat_history.insert(tk.END, f"Error: {e}\n\n", "error")
    
    chat_history.config(state=tk.DISABLED)
    chat_history.see(tk.END)
    
def save_transcript():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Note: Saving to Desktop for simplicity. Change if needed.
    filename = f"C:\\Users\\Denis\\Desktop\\chat_transcript_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Chat Transcript - {timestamp}\n\n")
        f.write(chat_history.get("1.0", tk.END))
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"Saved to {filename}\n\n", "system")

# --- GUI SETUP ---
root = tk.Tk()
root.title("Grok Chat (Vault Edition)")
root.geometry("700x500")

chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=25, bg="#f0f0f0")
chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_history.tag_config("user", foreground="blue")
chat_history.tag_config("grok", foreground="black")
chat_history.tag_config("error", foreground="red")
chat_history.tag_config("system", foreground="green")
chat_history.config(state=tk.DISABLED)

entry_frame = tk.Frame(root)
entry_frame.pack(padx=10, pady=5, fill=tk.X)

entry = Entry(entry_frame, width=60)
entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
entry.bind("<Return>", send_message)

send_button = Button(entry_frame, text="Send", command=send_message)
send_button.pack(side=tk.LEFT, padx=5)

save_button = Button(entry_frame, text="Save Chat", command=save_transcript)
save_button.pack(side=tk.LEFT, padx=5)

# --- START THE APP ---
# No more indexing needed here, just run the main loop.
root.mainloop()
