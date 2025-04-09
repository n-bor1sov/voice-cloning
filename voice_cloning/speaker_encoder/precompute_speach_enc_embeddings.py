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

sys.path.append('voice_cloning/textlesslib')

from textless.data.speech_encoder import SpeechEncoder

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dense_model_name = "mhubert-base-vp_mls_cv_8lang"
    quantizer_name, vocab_size = "kmeans", 2000
    model = SpeechEncoder.by_name(
        dense_model_name=dense_model_name,
        quantizer_model_name=quantizer_name,
        vocab_size=vocab_size,
        deduplicate=True,
        need_f0=False
    )
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
            embedding = model(waveform)  # Remove batch dimension and move to CPU
            # Store in dictionary with file path as key
            file_path = audio_paths[idx]
            embeddings_dict[file_path] = embedding

    # Save embeddings
    output_path = "checkpoints/precomputed_speach_enc_embeddings_libritts.pt"
    torch.save(embeddings_dict, output_path)
    print(f"Saved {len(embeddings_dict)} embeddings to {output_path}")

if __name__ == "__main__":
    main() 