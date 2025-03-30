import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.data.sampler import Sampler
import numpy as np
import torchaudio
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import os
import librosa

import sys
from pathlib import Path


# Auto-find project root (works across platforms)
if str(Path.cwd().parent) not in sys.path:
    sys.path.append(str(Path.cwd().parent))
    
# Assume ECAPA_TDNN model code is imported or defined here
from .ecapa_tdnn import ECAPA_TDNN_SMALL

class GE2ELoss(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
    
    def forward(self, embeddings):
        N, M, D = embeddings.shape
        
        centroids = torch.mean(embeddings, dim=1)
        sum_centroids = centroids * M
        sum_centroids_excl = sum_centroids.unsqueeze(1) - embeddings
        centroids_excl = sum_centroids_excl / (M - 1 + 1e-6)
        
        embeddings_flat = embeddings.reshape(N*M, D)
        centroids_excl_flat = centroids_excl.reshape(N*M, D)
        
        sim_matrix = F.cosine_similarity(
            embeddings_flat.unsqueeze(1), 
            centroids.unsqueeze(0), 
            dim=2
        )
        
        sim_self = F.cosine_similarity(embeddings_flat, centroids_excl_flat, dim=1)
        
        speaker_indices = torch.arange(N).view(N, 1).expand(-1, M).reshape(N*M)
        sim_matrix[torch.arange(N*M), speaker_indices] = sim_self
        
        sim_matrix = sim_matrix * self.w + self.b
        loss = F.cross_entropy(sim_matrix, speaker_indices.to(embeddings.device))
        
        return loss

class GE2EBatchSampler(Sampler):
    def __init__(self, dataset, n_speakers, n_utterances, num_batches):
        self.n_speakers = n_speakers
        self.n_utterances = n_utterances
        self.num_batches = num_batches
        
        self.speaker_to_indices = defaultdict(list)
        for idx, (_, spk_id) in enumerate(dataset):
            self.speaker_to_indices[spk_id].append(idx)
        
        self.speakers = list(self.speaker_to_indices.keys())
        
    def __iter__(self):
        for _ in range(self.num_batches):
            selected_speakers = np.random.choice(
                self.speakers, self.n_speakers, replace=False
            )
            batch = []
            for speaker in selected_speakers:
                indices = self.speaker_to_indices[speaker]
                if len(indices) < self.n_utterances:
                    selected = np.random.choice(
                        indices, self.n_utterances, replace=True
                    )
                else:
                    selected = np.random.choice(
                        indices, self.n_utterances, replace=False
                    )
                batch.extend(selected)
            yield batch
            
    def __len__(self):
        return self.num_batches

class VoxCeleb2Dataset(Dataset):
    def __init__(self, audio_paths, speaker_ids, sr=16000, duration=3):
        self.sr = sr
        self.duration = duration
        self.audio_paths = audio_paths
        self.speaker_ids = speaker_ids
        self.spk_to_id = {spk: idx for idx, spk in enumerate(set(speaker_ids))}
        
    def __len__(self):
        return len(self.audio_paths)
    
    def load_audio(self, path: str):
        wav_ref, sr = librosa.load(path)
        wav_ref = torch.FloatTensor(wav_ref).unsqueeze(0)
        resample_fn = torchaudio.transforms.Resample(sr, self.sr)
        wav_ref = resample_fn(wav_ref)
        return wav_ref
    
    def __getitem__(self, idx):
        # Load audio and process to fixed length
        waveform = self.load_audio(self.audio_paths[idx])  # Implement this
        waveform = self.process_waveform(waveform)
        speaker_id = self.spk_to_id[self.speaker_ids[idx]]
        return waveform, speaker_id
    
    def process_waveform(self, waveform):
        target_len = self.sr * self.duration
        if waveform.shape[-1] > target_len:
            start = np.random.randint(0, waveform.shape[-1] - target_len)
            waveform = waveform[..., start:start+target_len]
        else:
            pad = target_len - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad))
        return waveform.squeeze()

def collate_fn(batch):
    waveforms, speaker_ids = zip(*batch)
    waveforms = torch.stack(waveforms)
    speaker_ids = torch.LongTensor(speaker_ids)
    return waveforms, speaker_ids

# Training Configuration
n_speakers = 4      # Reduced from 5 to handle CPU memory better
n_utterances = 4    # Reduced from 5 to handle CPU memory better
batch_size = n_speakers * n_utterances  # = 16 utterances per batch
# Calculate a reasonable number of batches per epoch
# With 4874 recordings, let's aim to see each recording roughly once per epoch
num_batches = 4874 // batch_size  # ≈ 304 batches
emb_dim = 256
lr = 1e-4
num_epochs = 1

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# el
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

folder_path = "./voice_cloning/speaker_encoder/data/archive/vox1_test_wav/wav"

def get_audio_paths_and_speaker_ids(vox1_test_wav_folder):
    audio_paths = []
    speaker_ids = []

    # Add debug prints
    print(f"Looking for .wav files in: {os.path.abspath(vox1_test_wav_folder)}")

    # Traverse the directory structure
    for root, dirs, files in os.walk(vox1_test_wav_folder):
        for file in files:
            if file.endswith(".wav"):
                # Full path to the .wav file
                audio_paths.append(os.path.join(root, file))
                
                # Extract speaker ID from the path
                speaker_id = os.path.normpath(root).split(os.sep)[-2]
                speaker_ids.append(speaker_id)

    # Add debug prints
    print(f"Found {len(audio_paths)} audio files")
    print(f"Found {len(set(speaker_ids))} unique speakers")
    
    if len(audio_paths) == 0:
        raise ValueError(f"No .wav files found in {vox1_test_wav_folder}")

    return audio_paths, speaker_ids

def evaluate(model, test_csv_path, device):
    model.eval()
    test_df = pd.read_csv(test_csv_path)
    
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
folder_path = os.path.abspath(folder_path)

audio_paths, speaker_ids = get_audio_paths_and_speaker_ids(folder_path)

# Add debug print before creating dataset
print(f"Creating dataset with {len(audio_paths)} files and {len(set(speaker_ids))} speakers")

train_dataset = VoxCeleb2Dataset(audio_paths, speaker_ids)

# Add debug print for batch sampler
print(f"Number of speakers in dataset: {len(train_dataset.spk_to_id)}")

batch_sampler = GE2EBatchSampler(
    train_dataset, 
    n_speakers=min(n_speakers, len(train_dataset.spk_to_id)),  # Ensure n_speakers isn't larger than available speakers
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

test_csv_path = "./voice_cloning/speaker_encoder/data/archive/test.csv"

# Training Loop
best_accuracy = 0.0
for epoch in range(num_epochs):
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
    accuracy = evaluate(model, test_csv_path, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Evaluation Accuracy: {accuracy:.4f}")
    
    # Save checkpoint after each epoch
    checkpoint_dir = "checkpoints"
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