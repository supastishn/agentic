import os
import litellm
import tiktoken
import chromadb
import pathspec
from pathlib import Path
from rich.console import Console

console = Console()

# Files with these extensions will be indexed
INCLUDE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss", ".md", ".json", ".yml", ".yaml",
    ".java", ".cpp", ".c", ".h", ".cs", ".go", ".php", ".rb", ".rs", ".swift", ".kt", ".kts",
    "Dockerfile", ".sh", ".ps1", ".txt"
}

# These directories and files will be ignored by default, in addition to .gitignore rules
IGNORE_PATTERNS = {
    ".git", "__pycache__", "node_modules", "dist", "build", "target",
    ".venv", "venv", "env", ".env", "poetry.lock", "package-lock.json",
    ".agenticignore" # This file is for patterns, not to be indexed itself
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

    def __init__(self, project_path: str, config_dir: Path, embedding_config: dict, original_openai_api_base: str | None = None):
        self.project_path = Path(project_path)
        self.project_name = self.project_path.name
        self.embedding_config = embedding_config
        self.tokenizer = _get_tokenizer()
        self.original_openai_api_base = original_openai_api_base

        # Create a project-specific path for the RAG index to isolate databases
        safe_project_name = self.project_name.replace(".", "_").replace(os.sep, "_")
        db_path = config_dir / "data" / "rag_indices" / safe_project_name
        db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name="code-collection"
        )

    def has_index(self) -> bool:
        """Checks if the collection has any embeddings."""
        return self.collection.count() > 0

    def _scan_files(self) -> list[Path]:
        """Scans the project directory for files to index, respecting .gitignore and .agenticignore."""
        ignore_patterns = []
        # Add default ignore patterns
        ignore_patterns.extend(list(IGNORE_PATTERNS))
        
        gitignore_path = self.project_path / ".gitignore"
        agenticignore_path = self.project_path / ".agenticignore"

        if gitignore_path.is_file():
            with gitignore_path.open("r", encoding="utf-8") as f:
                ignore_patterns.extend(f.readlines())
        
        if agenticignore_path.is_file():
            with agenticignore_path.open("r", encoding="utf-8") as f:
                ignore_patterns.extend(f.readlines())

        spec = pathspec.PathSpec.from_lines('gitwildmatch', ignore_patterns)
        
        files_to_index = []
        for root, dirs, files in os.walk(self.project_path, topdown=True):
            root_path = Path(root)
            
            # Create relative paths for filtering
            rel_dirs = [root_path.joinpath(d).relative_to(self.project_path) for d in dirs]
            rel_files = [root_path.joinpath(f).relative_to(self.project_path) for f in files]

            # Filter directories in-place
            ignored_dirs = set(spec.match_files(rel_dirs))
            dirs[:] = [d for d, rel_d in zip(dirs, rel_dirs) if rel_d not in ignored_dirs]

            # Filter files
            ignored_files = set(spec.match_files(rel_files))
            for file_name, rel_file_path in zip(files, rel_files):
                if rel_file_path in ignored_files:
                    continue
                
                file_path = root_path / file_name
                if file_path.suffix in INCLUDE_EXTENSIONS or file_path.name in INCLUDE_EXTENSIONS:
                    files_to_index.append(file_path)

        return files_to_index

    def index_project(self, batch_size: int = 100, force_reindex: bool = False, quiet: bool = False):
        """Indexes all relevant files in the project."""
        if not force_reindex and self.has_index():
            if not quiet:
                console.print("[yellow]Existing RAG embeddings found. Skipping generation.[/yellow]")
                console.print("Use `/rag update` to force a re-index.")
            return

        if force_reindex and self.collection.count() > 0:
            if not quiet:
                console.print("[yellow]Forcing re-index. Clearing old embeddings...[/yellow]")
            self.client.delete_collection(name=self.collection.name)
            self.collection = self.client.create_collection(name=self.collection.name)

        files = self._scan_files()
        if not files:
            if not quiet:
                console.print("[yellow]No files found to index.[/yellow]")
            return
        
        if not quiet:
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
                if not quiet:
                    console.print(f"[yellow]Warning:[/] Could not read or process {file_path}: {e}")

        if not documents:
            if not quiet:
                console.print("[yellow]No content could be extracted from files.[/yellow]")
            return

        model = f"{self.embedding_config['provider']}/{self.embedding_config['model']}"
        api_key = self.embedding_config.get("api_key")

        # Process embeddings in batches to avoid API limits
        embeddings_list = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]

            def perform_embedding():
                # --- API base switching logic ---
                provider = self.embedding_config.get("provider")
                current_base = None
                if provider == 'openai':
                    current_base = os.environ.get("OPENAI_API_BASE")
                    if self.original_openai_api_base is not None:
                        os.environ["OPENAI_API_BASE"] = self.original_openai_api_base
                    elif "OPENAI_API_BASE" in os.environ:
                        del os.environ["OPENAI_API_BASE"]
                
                try:
                    response = litellm.embedding(
                        model=model,
                        input=batch_docs,
                        api_key=api_key
                    ).data
                finally:
                    if provider == 'openai':
                        if current_base is not None:
                            os.environ["OPENAI_API_BASE"] = current_base
                        elif "OPENAI_API_BASE" in os.environ:
                            del os.environ["OPENAI_API_BASE"]
                # --- End API base switching ---
                for item in response:
                    try:
                        # Try to access as an object attribute
                        embedding = item.embedding
                    except AttributeError:
                        # If that fails, try as a dictionary key
                        try:
                            embedding = item["embedding"]
                        except (KeyError, TypeError):
                            if not quiet:
                                console.print(
                                    "[bold red]Error: Could not find 'embedding' in the API response item.[/bold red]"
                                )
                                console.print("Offending item:")
                                console.print(item)
                            raise ValueError("Invalid embedding response format from API.")
                    embeddings_list.append(embedding)

            if quiet:
                perform_embedding()
            else:
                with console.status(f"[yellow]Generating embeddings for batch {i//batch_size + 1}...[/]"):
                    perform_embedding()

        # Add to ChromaDB in batches to avoid overwhelming the system
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings_list[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )
        if not quiet:
            console.print(f"[bold green]✔ Project indexed successfully.[/bold green] Total documents: {len(ids)}")

    def query(self, text: str, n_results=5) -> str:
        """Queries the RAG index and returns formatted context."""
        model = f"{self.embedding_config['provider']}/{self.embedding_config['model']}"
        api_key = self.embedding_config.get("api_key")
        
        # --- API base switching logic ---
        provider = self.embedding_config.get("provider")
        current_base = None
        if provider == 'openai':
            current_base = os.environ.get("OPENAI_API_BASE")
            if self.original_openai_api_base is not None:
                os.environ["OPENAI_API_BASE"] = self.original_openai_api_base
            elif "OPENAI_API_BASE" in os.environ:
                del os.environ["OPENAI_API_BASE"]

        try:
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

        finally:
            if provider == 'openai':
                if current_base is not None:
                    os.environ["OPENAI_API_BASE"] = current_base
                elif "OPENAI_API_BASE" in os.environ:
                    del os.environ["OPENAI_API_BASE"]
