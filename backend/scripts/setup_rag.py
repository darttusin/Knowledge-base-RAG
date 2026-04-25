"""Setup script for RAG system: index documents and train classifier."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag import Settings as RagSettings
from rag import create_embed_model
from rag.documents import load_documents, split_documents
from rag.vectorstore import create_collection, index_chunks


def index_documents():
    """Index documents into ChromaDB."""

    # Load settings
    settings = RagSettings()


    # Load documents
    docs = load_documents(settings.dataset_path)

    # Split documents
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)

    # Create embedding model
    embed_model = create_embed_model(settings)

    # Create collection
    collection = create_collection(settings.chroma_path, settings.chroma_collection)

    # Index chunks
    index_chunks(collection, chunks, embed_model, batch_size=200)



def train_classifier():
    """Train topic classifier for outlier detection."""
    from outlier_detection import TopicClassifier


    # Sample PyTorch-related questions for training
    pytorch_texts = [
        # Tensor operations
        "How to create a tensor in PyTorch?",
        "What is the difference between tensor and Tensor?",
        "How to convert numpy array to tensor?",
        "How to move tensor to GPU?",
        "What are tensor dimensions?",
        "How to reshape a tensor?",
        "How to concatenate tensors?",
        "What is tensor broadcasting?",

        # Neural networks
        "How to define a neural network in PyTorch?",
        "What is nn.Module?",
        "How to create a custom layer?",
        "What is forward() method?",
        "How to use nn.Sequential?",
        "What is the difference between nn.Linear and nn.Conv2d?",
        "How to implement a custom loss function?",

        # Training
        "How to train a model in PyTorch?",
        "What is autograd?",
        "How to compute gradients?",
        "What is backward() method?",
        "How to use optimizer?",
        "What is learning rate?",
        "How to implement training loop?",
        "How to save and load model?",

        # DataLoader
        "How to use DataLoader?",
        "What is Dataset class?",
        "How to create custom dataset?",
        "What is batch size?",
        "How to shuffle data?",
        "How to use data augmentation?",

        # Advanced topics
        "What is distributed training?",
        "How to use mixed precision?",
        "What is torch.cuda?",
        "How to profile PyTorch code?",
        "What is torch.jit?",
        "How to export model to ONNX?",
        "What is gradient clipping?",
        "How to implement attention mechanism?",

        # Common models
        "How to use pretrained models?",
        "What is torchvision?",
        "How to implement ResNet?",
        "What is transfer learning?",
        "How to fine-tune BERT?",

        # Loss functions
        "What is CrossEntropyLoss?",
        "How to use MSELoss?",
        "What is BCEWithLogitsLoss?",
        "How to implement custom loss?",

        # Optimization
        "What is Adam optimizer?",
        "How to use SGD?",
        "What is learning rate scheduling?",
        "How to implement early stopping?",

        # Debugging
        "How to debug PyTorch models?",
        "What is torch.no_grad()?",
        "How to check gradient flow?",
        "What is register_hook?",
    ]


    classifier = TopicClassifier(
        max_features=5000,
        nu=0.05,  # ~5% outliers
        kernel="rbf",
    )
    classifier.fit(pytorch_texts)

    # Test on some examples
    test_cases = [
        ("How to create a tensor?", True),
        ("What is the weather today?", False),
        ("How to use DataLoader in PyTorch?", True),
        ("Best restaurants in New York", False),
        ("What is autograd?", True),
        ("How to make pizza?", False),
    ]

    correct = 0
    for text, expected_on_topic in test_cases:
        result = classifier.predict(text)
        is_on_topic = result.labels[0] == 1
        abs(float(result.scores[0]))


        if is_on_topic == expected_on_topic:
            correct += 1

    correct / len(test_cases) * 100

    # Save classifier
    output_path = Path(__file__).parent.parent / "models" / "pytorch_classifier.joblib"
    output_path.parent.mkdir(exist_ok=True)

    classifier.save(output_path)



def main():
    """Main setup function."""
    import argparse

    parser = argparse.ArgumentParser(description="Setup RAG system")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip document indexing (use existing ChromaDB)"
    )
    parser.add_argument(
        "--skip-classifier",
        action="store_true",
        help="Skip classifier training"
    )

    args = parser.parse_args()


    if not args.skip_index:
        try:
            index_documents()
        except Exception:
            return 1
    else:
        pass

    if not args.skip_classifier:
        try:
            train_classifier()
        except Exception:
            return 1
    else:
        pass


    return 0


if __name__ == "__main__":
    sys.exit(main())
