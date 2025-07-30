from typing import List
import os
# We will use a transformer model for embeddings. In practice, this might be a separate service call.
try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    SentenceTransformer = None
    torch = None

class EmbeddingService:
    """Embedding service using BAAI/bge-base-en-v1.5 model to get text embeddings."""
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        if SentenceTransformer:
            # Determine the best device to use
            device = self._get_best_device()
            print(f"EmbeddingService: Using device: {device}")
            
            # Load the embedding model (this will download the model if not present)
            self.model = SentenceTransformer(model_name, device=device)
        else:
            self.model = None
            print("SentenceTransformer not installed. Embeddings will be dummy values.")

    def _get_best_device(self) -> str:
        """Determine the best device to use for embeddings."""
        # Check environment variable for forced device
        forced_device = os.getenv("EMBEDDING_DEVICE")
        if forced_device and forced_device.lower() != 'auto':
            return forced_device.lower()
            
        # If torch is available, check CUDA compatibility
        if torch and torch.cuda.is_available():
            try:
                # More thorough CUDA test - try actual embedding operations
                test_tensor = torch.zeros(1, 10).cuda()  # Simulate embedding tensor
                test_output = torch.nn.functional.embedding(
                    torch.tensor([[0, 1]], device='cuda'), 
                    test_tensor
                )
                del test_tensor, test_output
                torch.cuda.empty_cache()
                return 'cuda'
            except Exception as e:
                print(f"CUDA available but not compatible with current hardware: {e}")
                print("Falling back to CPU for embeddings")
                return 'cpu'
        else:
            return 'cpu'

    def embed(self, text: str) -> List[float]:
        """Convert text into a vector embedding."""
        if self.model:
            embedding: List[float] = self.model.encode(text, show_progress_bar=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        # Fallback: return a dummy embedding (e.g., vector of zeros) if model not available
        return [0.0] * 768  # assuming 768-dim for BGE base model
