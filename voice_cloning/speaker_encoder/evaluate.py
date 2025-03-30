import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
import torchaudio
import librosa
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Try Importing the Model ---
try:
    from voice_cloning.speaker_encoder.ecapa_tdnn import ECAPA_TDNN_SMALL
except ImportError:
    print("Error: Could not import ECAPA_TDNN_SMALL.")
    print(f"Looked in project root: {project_root}")
    print("Please ensure the 'voice_cloning' directory exists at the project root,")
    print("or that the package is correctly installed/discoverable.")
    sys.exit(1)

# --- Helper Functions ---

def load_audio(filepath, target_sr=16000):
    """Loads and preprocesses an audio file."""
    try:
        wav, sr = librosa.load(filepath, sr=None) # Load with original sample rate
        if wav.ndim > 1: # Convert to mono if stereo
            # Use average for stereo to mono conversion
            wav = np.mean(wav, axis=0)
        if sr != target_sr:
            # Use torchaudio for resampling tensor
            wav_tensor = torch.FloatTensor(wav).unsqueeze(0) # Add channel dim if needed by resample
            resample_fn = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wav = resample_fn(wav_tensor).squeeze(0).numpy() # Remove channel dim

        # Ensure it's a 1D array before making tensor
        if wav.ndim > 1:
            wav = wav.squeeze()

        return torch.FloatTensor(wav).unsqueeze(0) # Add batch dimension
    except Exception as e:
        print(f"Error loading or processing file {filepath}: {e}")
        return None

@torch.no_grad()
def get_embedding(wav_tensor, model, device):
    """Extracts speaker embedding using the model."""
    if wav_tensor is None:
        return None
    try:
        wav_tensor = wav_tensor.to(device)
        embedding = model(wav_tensor)
        embedding = embedding / embedding.norm() # Normalize
        return embedding.cpu() # Move embedding back to CPU for calculations
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def calculate_eer(y_true, y_scores):
    """
    Calculates the Equal Error Rate (EER) and the threshold at EER.
    Lower score indicates non-match (different speaker), higher score indicates match (same speaker).
    y_scores should represent the likelihood of being the *same* speaker.
    """
    try:
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        from sklearn.metrics import roc_curve
    except ImportError:
        print("Error: EER calculation requires SciPy and scikit-learn.")
        print("Please install them: pip install scipy scikit-learn")
        return None, None

    try:
        # Ensure scores are numpy array
        y_scores = np.asarray(y_scores)
        y_true = np.asarray(y_true)

        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

        # EER is where False Rejection Rate (FRR = 1 - TPR) equals False Acceptance Rate (FPR)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        # Find the threshold corresponding to EER
        # Need to interpolate from FPR values to the actual score thresholds
        eer_threshold = float(interp1d(fpr, thresholds)(eer))

        return eer, eer_threshold
    except Exception as e:
        print(f"Error calculating EER: {e}")
        return None, None


# --- Main Script ---

def main(args):
    # Determine device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Loading speaker encoder model from: {args.checkpoint_path}")
    try:
        spk_embedder = ECAPA_TDNN_SMALL(
            feat_dim=1024,
            feat_type="fbank", # Assuming fbank based on notebook context
        )

        # Load checkpoint, handling potential CPU/GPU mapping issues
        checkpoint = torch.load(args.checkpoint_path, map_location=device)

        # Determine the key for the state dictionary
        state_dict_key = None
        if 'model' in checkpoint:
            state_dict_key = 'model'
        elif 'state_dict' in checkpoint:
            state_dict_key = 'state_dict'

        if state_dict_key:
            # Adjust keys if necessary (e.g., remove 'module.' prefix if saved with DataParallel)
            state_dict = checkpoint[state_dict_key]
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k # remove `module.` prefix
                new_state_dict[name] = v
            spk_embedder.load_state_dict(new_state_dict, strict=True)
        else:
            # Try loading directly if no common key found
            print("Warning: Could not find 'model' or 'state_dict' key in checkpoint. Attempting to load root.")
            spk_embedder.load_state_dict(checkpoint, strict=True)

        spk_embedder.to(device)
        spk_embedder.eval()
        print("Speaker encoder model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        sys.exit(1)

    # --- Load Test Data ---
    print(f"Loading test data from: {args.test_csv_path}")
    test_csv_path = Path(args.test_csv_path).resolve()
    csv_dir = test_csv_path.parent
    try:
        test_df = pd.read_csv(test_csv_path)
        # --- Basic Validation of CSV Structure ---
        required_columns = ['audio_path1', 'audio_path2', 'label']
        if not all(col in test_df.columns for col in required_columns):
            print(f"Error: CSV file '{test_csv_path}' must contain columns: {required_columns}")
            sys.exit(1)
        print(f"Found {len(test_df)} pairs in the test file.")
    except FileNotFoundError:
        print(f"Error: Test CSV file not found at {test_csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading test CSV file: {e}")
        sys.exit(1)

    # --- Process Pairs and Calculate Scores ---
    results = []
    print("Processing audio pairs and calculating similarities...")

    for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # Resolve paths: Try relative to project root first, then relative to CSV directory
        path1_rel = row['audio_path1']
        path2_rel = row['audio_path2']

        path1_abs_root = project_root / path1_rel
        path2_abs_root = project_root / path2_rel

        path1_abs_csv = (csv_dir / path1_rel).resolve()
        path2_abs_csv = (csv_dir / path2_rel).resolve()

        if path1_abs_root.exists():
            path1 = path1_abs_root
        elif path1_abs_csv.exists():
            path1 = path1_abs_csv
        else:
            print(f"Warning: Skipping pair {index}. Cannot find file for '{path1_rel}' (checked rel to root and CSV).")
            continue

        if path2_abs_root.exists():
            path2 = path2_abs_root
        elif path2_abs_csv.exists():
            path2 = path2_abs_csv
        else:
            print(f"Warning: Skipping pair {index}. Cannot find file for '{path2_rel}' (checked rel to root and CSV).")
            continue


        wav1 = load_audio(str(path1))
        wav2 = load_audio(str(path2))

        emb1 = get_embedding(wav1, spk_embedder, device)
        emb2 = get_embedding(wav2, spk_embedder, device)

        if emb1 is not None and emb2 is not None:
            # Cosine similarity: Higher score means more similar (closer to 1)
            similarity = F.cosine_similarity(emb1, emb2).item()
            results.append({
                'pair_index': index,
                'label': int(row['label']), # Ground truth label (1=same, 0=different)
                'similarity_score': similarity   # Score used for evaluation
            })
        else:
            print(f"Warning: Skipping pair {index} due to audio processing or embedding errors.")

    if not results:
        print("\nError: No pairs could be processed successfully.")
        sys.exit(1)

    results_df = pd.DataFrame(results)
    print("\n--- Evaluation ---")

    # --- Calculate EER ---
    # EER calculation expects scores where higher means more likely to be positive class (label=1)
    # Cosine similarity fits this: higher similarity -> more likely same speaker.
    true_labels = results_df['label'].values
    similarity_scores = results_df['similarity_score'].values

    eer, eer_threshold = calculate_eer(true_labels, similarity_scores)

    if eer is not None and eer_threshold is not None:
        print(f"\nEqual Error Rate (EER): {eer:.4f}")
        print(f"Threshold at EER (Cosine Similarity): {eer_threshold:.4f}")
        # Distance = 1 - Similarity. Threshold for distance would be 1 - eer_threshold
        print(f"(Equivalent Distance Threshold: {1 - eer_threshold:.4f})")
    else:
        print("\nCould not compute EER.")

    # --- Optional: Save detailed results ---
    if args.output_csv:
        output_path = Path(args.output_csv).resolve()
        print(f"\nSaving detailed similarity scores to: {output_path}")
        # Add back file paths if needed, handle potential length issues for CSV
        full_results_df = pd.merge(test_df.iloc[results_df['pair_index']], results_df, on='pair_index')
        full_results_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Speaker Encoder Checkpoint using a test CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("checkpoint_path", type=str,
                        help="Path to the speaker encoder .pt checkpoint file.")
    parser.add_argument("test_csv_path", type=str,
                        help="Path to the test CSV file. Required columns: 'audio_path1', 'audio_path2', 'label' (1 or 0).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on ('cuda' or 'cpu'). Will use CPU if CUDA is not available.")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional path to save a CSV file with detailed similarity scores for each pair.")

    args = parser.parse_args()

    # Basic checks for file existence before starting
    if not Path(args.checkpoint_path).is_file():
        print(f"Error: Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    if not Path(args.test_csv_path).is_file():
        print(f"Error: Test CSV file not found: {args.test_csv_path}")
        sys.exit(1)

    main(args)