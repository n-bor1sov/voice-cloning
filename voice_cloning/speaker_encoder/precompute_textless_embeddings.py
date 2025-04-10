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
sys.path.append("voice_cloning/textlesslib")


from voice_cloning.textlesslib.textless.data.speech_encoder import SpeechEncoder

def process_unit(encoded, sampling_rate, hop_length):
    # A method that aligns units and durations (50Hz) extracted from 16kHz audio with
    # mel-spectrograms extracted from 22,050Hz audio.

    unit = encoded["units"].cpu().tolist()
    duration = encoded["durations"].cpu().tolist()

    duration = [int(i) * (sampling_rate // 50) for i in duration]

    expand_unit = []

    for u, d in zip(unit, duration):
        for _ in range(d):
            expand_unit.append(u)

    new_length = len(expand_unit) // hop_length * hop_length

    unit = torch.LongTensor(expand_unit)[:new_length].reshape(-1, hop_length).mode(1)[0].tolist()

    squeezed_unit = [unit[0]]
    squeezed_duration = [1]

    for u in unit[1:]:
        if u == squeezed_unit[-1]:
            squeezed_duration[-1] += 1
        else:
            squeezed_unit.append(u)
            squeezed_duration.append(1)

    unit = torch.LongTensor(squeezed_unit)
    duration = torch.LongTensor(squeezed_duration)

    return unit, duration

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained ECAPA model
    # checkpoint_path = "voice_cloning/Grad-TTS/checkpts/speaker_encoder.pt"
    # if not os.path.isfile(checkpoint_path):
    #     raise FileNotFoundError(f"Pre-trained model not found at {checkpoint_path}")

    print("Loading pre-trained ECAPA model...")
    dense_model_name = "mhubert-base-vp_en_es_fr"
    quantizer_name, vocab_size = "kmeans", 1000

    unit_extractor = SpeechEncoder.by_name(
        dense_model_name=dense_model_name,
        quantizer_model_name=quantizer_name,
        vocab_size=vocab_size,
        deduplicate=True,
        need_f0=False
    )
    _ = unit_extractor.cuda().eval()

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
            encoded = unit_extractor(waveform)
            unit, duration = process_unit(encoded, 24000, 256)
            
            # Store in dictionary with file path as key
            file_path = audio_paths[idx]
            embeddings_dict[file_path] = (unit, duration)

    # Save embeddings
    os.makedirs("checkpoints", exist_ok=True)
    output_path = "checkpoints/precomputed_textless_embeddings_libritts.pt"
    torch.save(embeddings_dict, output_path)
    print(f"Saved {len(embeddings_dict)} embeddings to {output_path}")

if __name__ == "__main__":
    main() 