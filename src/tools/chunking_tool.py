import math

class ChunkingTool:
    """Utility to split text into chunks for processing (e.g., indexing to vector DB)."""
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size  # default chunk size in characters (or tokens approx.)

    def chunk(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list:
        """Split a long text into chunks with optional overlap."""
        chunks = []
        length = len(text)
        step = chunk_size - overlap
        
        for i in range(0, length, step):
            chunk = text[i: i + chunk_size]
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            # Stop if we've reached the end
            if i + chunk_size >= length:
                break
                
        return chunks

    def chunk_text(self, text: str) -> list:
        """Split a long text into chunks of approximately chunk_size characters."""
        return self.chunk(text, self.chunk_size, 0)

    def chunk_file(self, file_path: str) -> list:
        """Read a file and split its content into chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
        return self.chunk_text(content)
