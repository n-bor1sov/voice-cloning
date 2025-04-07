""" from https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification """

import torch
import fairseq
from packaging import version
import torch.nn.functional as F
from fairseq import tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from omegaconf import OmegaConf
from s3prl.upstream.interfaces import UpstreamBase
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import librosa
import torchaudio
import os

def load_model(filepath):
    state = torch.load(filepath, map_location=lambda storage, loc: storage)

    state["cfg"] = OmegaConf.create(state["cfg"])

    if "args" in state and state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    elif "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
    else:
        raise RuntimeError(
            f"Neither args nor cfg exist in state keys = {state.keys()}"
            )

    task = tasks.setup_task(cfg.task)
    if "task_state" in state:
        task.load_state_dict(state["task_state"])

    model = task.build_model(cfg.model)

    return model, cfg, task


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        assert version.parse(fairseq.__version__) > version.parse(
            "0.10.2"
        ), "Please install the fairseq master branch."

        model, cfg, task = load_model(ckpt)
        self.model = model
        self.task = task

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

    def forward(self, wavs):
        if self.task.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )
        return {
            "default": features,
        }

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

class PrecomputedEmbeddingsDataset(Dataset):
    def __init__(self, embeddings_dict, speaker_ids):
        self.embeddings_dict = embeddings_dict
        self.audio_paths = list(embeddings_dict.keys())
        self.speaker_ids = speaker_ids
        self.spk_to_id = {spk: idx for idx, spk in enumerate(set(speaker_ids))}
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        file_path = self.audio_paths[idx]
        embedding = self.embeddings_dict[file_path]
        speaker_id = self.spk_to_id[self.speaker_ids[idx]]
        return embedding, speaker_id