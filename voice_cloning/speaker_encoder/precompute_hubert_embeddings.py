import torch
import torch.nn as nn
import torchaudio
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
from .utils import VoxCeleb2Dataset, get_audio_paths_and_speaker_ids
import sys
import torch.nn.functional as F

# Add project root to path
if str(Path.cwd().parent) not in sys.path:
    sys.path.append(str(Path.cwd().parent))

from transformers import HubertModel

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        self.final_proj = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained ECAPA model
    checkpoint_path = "voice_cloning/Grad-TTS/checkpts/speaker_encoder.pt"
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Pre-trained model not found at {checkpoint_path}")

    print("Loading pre-trained ECAPA model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = HubertModelWithFinalProj.from_pretrained("lengyue233/content-vec-best")
    _ = model.cuda().eval()
    print("Model loaded successfully")

    # Get all audio files
    folder_path = "./text_encoder_dataset"
    audio_paths, speaker_ids = get_audio_paths_and_speaker_ids(folder_path)

    # Create dataset
    dataset = VoxCeleb2Dataset(audio_paths, speaker_ids)

    embeddings_dict = {}
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Computing embeddings"):
            waveform, speaker_id = dataset[idx]
            waveform = waveform.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get embedding
            embedding = model(waveform)["last_hidden_state"]
            embedding = embedding.squeeze(0).cpu()  # Remove batch dimension and move to CPU
            
            # Store in dictionary with file path as key
            file_path = audio_paths[idx]
            embeddings_dict[file_path] = embedding

    # Save embeddings
    output_path = "checkpoints/precomputed_hubert_embeddings_libritts.pt"
    torch.save(embeddings_dict, output_path)
    print(f"Saved {len(embeddings_dict)} embeddings to {output_path}")

if __name__ == "__main__":
    main() 