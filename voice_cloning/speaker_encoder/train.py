import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.data.sampler import Sampler
import numpy as np
from collections import defaultdict

# Assume ECAPA_TDNN model code is imported or defined here
from ecapa_tdnn import ECAPA_TDNN_SMALL

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
n_speakers = 64
n_utterances = 10
batch_size = n_speakers * n_utterances
num_batches = 1000  # Adjust based on dataset size
emb_dim = 192
lr = 1e-4
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dataset = VoxCeleb2Dataset(audio_paths, speaker_ids)
batch_sampler = GE2EBatchSampler(
    train_dataset, n_speakers, n_utterances, num_batches
)
train_loader = DataLoader(
    train_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn
)

# Model and Loss
model = ECAPA_TDNN_SMALL(feat_dim=80, emb_dim=emb_dim).to(device)
criterion = GE2ELoss().to(device)
optimizer = Adam(model.parameters(), lr=lr)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (waveforms, _) in enumerate(train_loader):
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