import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
from tqdm import tqdm
from .utils import GE2ELoss, GE2EBatchSampler, collate_fn, get_audio_paths_and_speaker_ids, PrecomputedEmbeddingsDataset
import pandas as pd

def evaluate_linear(linear_layer, embeddings_dict, test_csv_path, device, folder_path):
    linear_layer.eval()
    test_df = pd.read_csv(test_csv_path).sample(n=500, random_state=42)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            audio1_path = os.path.join(folder_path, row['audio_1'])
            audio2_path = os.path.join(folder_path, row['audio_2'])
            
            # Get pre-computed embeddings
            emb1 = embeddings_dict[audio1_path].to(device)
            emb2 = embeddings_dict[audio2_path].to(device)
            
            # Pass through linear layer
            emb1 = linear_layer(emb1)
            emb2 = linear_layer(emb2)
            
            # Calculate similarity
            similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
            
            # Predict (similarity > 0.5 indicates same speaker)
            prediction = (similarity > 0.5).int().item()
            
            # Compare with ground truth
            correct += (prediction == row['label'])
            total += 1
    
    accuracy = correct / total
    linear_layer.train()
    return accuracy

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-computed embeddings
    embeddings_path = "checkpoints/speaker_encoder/precomputed_embeddings.pt"
    if not os.path.isfile(embeddings_path):
        raise FileNotFoundError(f"Pre-computed embeddings not found at {embeddings_path}. Run precompute_embeddings.py first.")

    embeddings_dict = torch.load(embeddings_path)
    print(f"Loaded {len(embeddings_dict)} embeddings")

    # Get speaker IDs
    folder_path = "./voice_cloning/speaker_encoder/data/archive/vox1_test_wav/wav"
    audio_paths, speaker_ids = get_audio_paths_and_speaker_ids(folder_path)

    # Create dataset
    dataset = PrecomputedEmbeddingsDataset(embeddings_dict, speaker_ids)

    # Training Configuration
    n_speakers = 4
    n_utterances = 4
    batch_size = n_speakers * n_utterances
    num_batches = len(dataset) // batch_size
    input_dim = 256
    output_dim = 64
    lr = 1e-4
    num_epochs = 50

    # Initialize linear layer
    linear_layer = nn.Linear(input_dim, output_dim).to(device)
    criterion = GE2ELoss().to(device)
    optimizer = Adam(linear_layer.parameters(), lr=lr)

    # Create dataloader
    batch_sampler = GE2EBatchSampler(
        dataset,
        n_speakers=min(n_speakers, len(dataset.spk_to_id)),
        n_utterances=n_utterances,
        num_batches=num_batches
    )
    train_loader = DataLoader(
        dataset, batch_sampler=batch_sampler, collate_fn=collate_fn
    )

    # Training loop
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        linear_layer.train()
        total_loss = 0.0

        for batch_idx, (embeddings, _) in tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}"):
            embeddings = embeddings.to(device)
            
            # Pass through linear layer
            transformed_embeddings = linear_layer(embeddings)
            
            # Reshape for GE2ELoss
            transformed_embeddings = transformed_embeddings.view(n_speakers, n_utterances, -1)
            
            # Calculate loss
            loss = criterion(transformed_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Evaluate
        test_csv_path = "./voice_cloning/speaker_encoder/data/archive/test.csv"
        accuracy = evaluate_linear(linear_layer, embeddings_dict, test_csv_path, device, folder_path)
        print(f"Epoch [{epoch+1}/{num_epochs}], Evaluation Accuracy: {accuracy:.4f}")

        # Save checkpoint
        checkpoint_dir = "checkpoints/speaker_encoder"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch + 1,
            'linear_state_dict': linear_layer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'accuracy': accuracy
        }

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint_path = os.path.join(checkpoint_dir, 'best_linear_model.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"New best model saved with accuracy: {accuracy:.4f}")

        # Save latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_linear_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main() 