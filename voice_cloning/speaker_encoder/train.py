import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
import os
from .utils import GE2ELoss, GE2EBatchSampler, VoxCeleb2Dataset, collate_fn, get_audio_paths_and_speaker_ids
import sys
from pathlib import Path

if str(Path.cwd().parent) not in sys.path:
    sys.path.append(str(Path.cwd().parent))

from .ecapa_tdnn import ECAPA_TDNN_SMALL

# Training Configuration
n_speakers = 4
n_utterances = 4
batch_size = n_speakers * n_utterances
num_batches = 4874 // batch_size
emb_dim = 256
lr = 1e-5
num_epochs = 60

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def evaluate(model, test_csv, device, folder_path):
    model.eval()
    test_df = pd.read_csv(test_csv).sample(n=500, random_state=42)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            # Load both audio files
            audio1_path = os.path.join(folder_path, row['audio_1'])
            audio2_path = os.path.join(folder_path, row['audio_2'])
            
            # Create dataset instances for single files
            dataset = VoxCeleb2Dataset([audio1_path, audio2_path], ['spk1', 'spk2'])
            
            # Get embeddings
            audio1, _ = dataset[0]
            audio2, _ = dataset[1]
            
            audio1 = audio1.unsqueeze(0).to(device)
            audio2 = audio2.unsqueeze(0).to(device)
            
            emb1 = model(audio1)
            emb2 = model(audio2)
            
            # Calculate similarity
            similarity = F.cosine_similarity(emb1, emb2)
            
            # Predict (similarity > 0.5 indicates same speaker)
            prediction = (similarity > 0.5).int().item()
            
            # Compare with ground truth
            correct += (prediction == row['label'])
            total += 1
    
    accuracy = correct / total
    return accuracy

# Make folder path absolute if it's relative
folder_path = "./voice_cloning/speaker_encoder/data/train/vox1_test_wav/wav"
test_folder_path = "./voice_cloning/speaker_encoder/data/test/sample"
folder_path = os.path.abspath(folder_path)

# Extract data from folder and create a dataset
audio_paths, speaker_ids = get_audio_paths_and_speaker_ids(folder_path)
train_dataset = VoxCeleb2Dataset(audio_paths, speaker_ids)

# Add debug print for batch sampler
print(f"Number of speakers in dataset: {len(train_dataset.spk_to_id)}")

batch_sampler = GE2EBatchSampler(
    train_dataset, 
    n_speakers=min(n_speakers, len(train_dataset.spk_to_id)),
    n_utterances=n_utterances, 
    num_batches=num_batches
)

train_loader = DataLoader(
    train_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn
)

# Model and Loss
model = ECAPA_TDNN_SMALL(feat_dim=256, emb_dim=emb_dim).to(device)
criterion = GE2ELoss().to(device)
optimizer = Adam(model.parameters(), lr=lr)

train_csv = "./voice_cloning/speaker_encoder/data/train/test.csv"
test_csv = "./voice_cloning/speaker_encoder/data/test/test.csv"
resume_checkpoint = "checkpoints/speaker_encoder/best_model.pt" # or None if we want to train from scratch

start_epoch = 0
best_accuracy = 0.0

if resume_checkpoint and os.path.isfile(resume_checkpoint):
    print(f"Resuming training from checkpoint: {resume_checkpoint}")
    checkpoint = torch.load(resume_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint.get('accuracy', 0.0) # Use get for backward compatibility
    print(f"Loaded checkpoint from epoch {start_epoch} with best accuracy {best_accuracy:.4f}")
else:
    print("Starting training from scratch.")

# Training Loop
print(f"Starting training loop from epoch {start_epoch + 1}")
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (waveforms, _) in tqdm(enumerate(train_loader), total=num_batches):
        waveforms = waveforms.to(device)
        
        embeddings = model(waveforms)
        embeddings = embeddings.view(n_speakers, n_utterances, -1)
        
        loss = criterion(embeddings)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    
    # Evaluate after each epoch
    accuracy = evaluate(model, test_csv, device, folder_path=folder_path)
    print(f"Epoch [{epoch+1}/{num_epochs}], Evaluation Accuracy: {accuracy:.4f}")
    
    checkpoint_dir = "checkpoints/speaker_encoder"
    # Save checkpoint after each epoch
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'accuracy': accuracy
    }
    
    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"New best model saved with accuracy: {accuracy:.4f}")
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")