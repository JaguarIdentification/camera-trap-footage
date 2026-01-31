import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for pre-computed embeddings."""
    
    def __init__(self, embeddings: list[list[float]], labels: list[int]) -> None:
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the embedding and label for a given index."""
        return self.embeddings[idx], self.labels[idx]
