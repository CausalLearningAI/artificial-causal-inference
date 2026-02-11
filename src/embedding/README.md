# Embedding Extraction

Extract embeddings from frames → `dataset/{subject}/{version}/embeddings/full/{encoder}/{token}/`

## Extract

```bash
python src/embedding/extract.py experiment=ants/v1 encoder=dinov2
```

Python:
```python
from src.dataset.get_dataset import load_dataset
from src.embedding import extract_embeddings_to_disk, load_embeddings_from_disk, load_dataset_with_embeddings

dataset = load_dataset(subject='ants', version='v1')
extract_embeddings_to_disk(dataset, encoder='dinov2')

# Later
emb = load_embeddings_from_disk('ants', 'v1', encoder='dinov2')

# Or load dataset with embeddings
dataset = load_dataset_with_embeddings('ants', 'v1', encoder='dinov2')
```

## Encoders

| Model | Dims | Speed |
|-------|------|-------|
| **dinov2** | 768 | 12ms/img |
| siglip | 1024 | 14ms/img |
| clip | 512 | 15ms/img |
| clip_large | 768 | 30ms/img |
| vit | 768 | 13ms/img |
| mae | 1024 | 20ms/img |
| dinov2_large | 1024 | 25ms/img |
| resnet | 2048 | 8ms/img |

## API

`extract_embeddings_to_disk(dataset, encoder='dinov2', token='class', batch_size=32, num_workers=4, device='cuda', force=False, ...)`

Save embeddings to disk.

`load_embeddings_from_disk(subject, version, encoder='dinov2', token='class', dataset_root='./dataset')`

Load saved embeddings → `torch.Tensor(num_samples, embedding_dim)`

## Token Types

- **'class'** (default): CLS token, fast & compact
- **'mean'**: Mean pooling, preserves spatial info

Both same dimensionality.
