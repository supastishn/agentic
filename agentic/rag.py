import os
import litellm
import tiktoken
import chromadb
from pathlib import Path
from rich.console import Console

console = Console()

# Files with these extensions will be indexed
INCLUDE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss", ".md", ".json", ".yml", ".yaml",
    ".java", ".cpp", ".c", ".h", ".cs", ".go", ".php", ".rb", ".rs", ".swift", ".kt", ".kts",
    "Dockerfile", ".sh", ".ps1", ".txt"
}

# These directories and files will be ignored
IGNORE_PATTERNS = {
    ".git", "__pycache__", "node_modules", "dist", "build", "target",
    ".venv", "venv", "env", ".env", "poetry.lock", "package-lock.json"
}

def _get_tokenizer():
    """Get a tokenizer for splitting text."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return tiktoken.get_encoding("gpt2") # Fallback

def _split_text(text: str, tokenizer, chunk_size=1000, chunk_overlap=100):
    """Splits text into chunks using a tokenizer."""
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

class CodeRAG:
    """Manages the Retrieval-Augmented Generation for a code project."""

    def __init__(self, project_path: str, config_dir: Path, embedding_config: dict):
        self.project_path = Path(project_path)
        self.project_name = self.project_path.name
        self.embedding_config = embedding_config
        self.tokenizer = _get_tokenizer()

        # Create a project-specific path for the RAG index to isolate databases
        safe_project_name = self.project_name.replace(".", "_").replace(os.sep, "_")
        db_path = config_dir / "data" / "rag_indices" / safe_project_name
        db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name="code-collection"
        )

    def _scan_files(self) -> list[Path]:
        """Scans the project directory for files to index."""
        files_to_index = []
        for root, dirs, files in os.walk(self.project_path):
            # Remove ignored directories from traversal
            dirs[:] = [d for d in dirs if d not in IGNORE_PATTERNS]
            
            for file in files:
                if file in IGNORE_PATTERNS:
                    continue
                file_path = Path(root) / file
                if file_path.suffix in INCLUDE_EXTENSIONS or file_path.name in INCLUDE_EXTENSIONS:
                    files_to_index.append(file_path)
        return files_to_index

    def index_project(self, batch_size: int = 100, force_reindex: bool = False):
        """Indexes all relevant files in the project."""
        if not force_reindex and self.collection.count() > 0:
            console.print("[yellow]Existing RAG embeddings found. Skipping generation.[/yellow]")
            console.print("Use `/rag update` to force a re-index.")
            return

        if force_reindex and self.collection.count() > 0:
            console.print("[yellow]Forcing re-index. Clearing old embeddings...[/yellow]")
            self.client.delete_collection(name=self.collection.name)
            self.collection = self.client.create_collection(name=self.collection.name)

        files = self._scan_files()
        if not files:
            console.print("[yellow]No files found to index.[/yellow]")
            return
        
        console.print(f"Found {len(files)} files to index. Generating embeddings...")

        documents = []
        metadatas = []
        ids = []
        doc_id_counter = 0

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                relative_path = file_path.relative_to(self.project_path)
                chunks = _split_text(content, self.tokenizer)
                
                for chunk in chunks:
                    documents.append(chunk)
                    metadatas.append({"source": str(relative_path)})
                    ids.append(f"doc_{doc_id_counter}")
                    doc_id_counter += 1
            except Exception as e:
                console.print(f"[yellow]Warning:[/] Could not read or process {file_path}: {e}")

        if not documents:
            console.print("[yellow]No content could be extracted from files.[/yellow]")
            return

        model = f"{self.embedding_config['provider']}/{self.embedding_config['model']}"
        api_key = self.embedding_config.get("api_key")

        # Process embeddings in batches to avoid API limits
        embeddings_list = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            with console.status(f"[yellow]Generating embeddings for batch {i//batch_size + 1}...[/]"):
                response = litellm.embedding(
                    model=model,
                    input=batch_docs,
                    api_key=api_key
                ).data
                for item in response:
                    try:
                        # Try to access as an object attribute
                        embedding = item.embedding
                    except AttributeError:
                        # If that fails, try as a dictionary key
                        try:
                            embedding = item["embedding"]
                        except (KeyError, TypeError):
                            console.print(
                                "[bold red]Error: Could not find 'embedding' in the API response item.[/bold red]"
                            )
                            console.print("Offending item:")
                            console.print(item)
                            raise ValueError("Invalid embedding response format from API.")
                    embeddings_list.append(embedding)

        # Add to ChromaDB in batches to avoid overwhelming the system
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings_list[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )
        console.print(f"[bold green]✔ Project indexed successfully.[/bold green] Total documents: {len(ids)}")

    def query(self, text: str, n_results=5) -> str:
        """Queries the RAG index and returns formatted context."""
        model = f"{self.embedding_config['provider']}/{self.embedding_config['model']}"
        api_key = self.embedding_config.get("api_key")
        
        response_item = litellm.embedding(model=model, input=[text], api_key=api_key).data[0]
        
        try:
            # Try to access as an object attribute
            query_embedding = response_item.embedding
        except AttributeError:
            # If that fails, try as a dictionary key
            try:
                query_embedding = response_item["embedding"]
            except (KeyError, TypeError):
                console.print(
                    "[bold red]Error: Could not find 'embedding' in the query response item.[/bold red]"
                )
                console.print("Offending item:")
                console.print(response_item)
                raise ValueError("Invalid embedding response format from API for query.")
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        context_parts = []
        sources_seen = set()
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                source = results["metadatas"][0][i]["source"]
                if source not in sources_seen:
                    context_parts.append(f"----- From: {source} -----")
                    sources_seen.add(source)
                context_parts.append(doc)
        
        return "\n\n".join(context_parts) if context_parts else "No relevant context found in the index."
