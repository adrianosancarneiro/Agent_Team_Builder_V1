import math

class ChunkingTool:
    """Utility to split text into chunks for processing (e.g., indexing to vector DB)."""
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size  # default chunk size in characters (or tokens approx.)

    def chunk_text(self, text: str) -> list:
        """Split a long text into chunks of approximately chunk_size characters."""
        chunks = []
        length = len(text)
        for i in range(0, length, self.chunk_size):
            chunk = text[i: i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def chunk_file(self, file_path: str) -> list:
        """Read a file and split its content into chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
        return self.chunk_text(content)
