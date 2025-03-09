# Voice Cloning Project

## Description
This project aims to develop a voice cloning system that can synthesize speech in the voice of a target speaker using only a small amount of reference audio. The system leverages state-of-the-art deep learning techniques to create natural-sounding voice clones.

## Usage
1. **Speaker Encoder Training**:
   ```
   python -m voice_cloning.speaker_encoder.train
   ```

2. **Notebooks**:
   - See the `notebooks/` directory for example implementations of different components
    - **`notebooks/speaker_encoder.ipynb` is the simpel example how to test speaker encoder on a few samples**

## What Has Been Done So Far

- We performed a small research in the field of voice cloning and speech synthesis to identify the most effective approaches
- We wrote code for training ECAPA-TDNN speaker encoder on the VoxCeleb dataset. Implemented a GE2E loss function.
- Created example notebooks for different parts of the pipeline:
  - Speaker encoder training and inference
  - HuBERT content encoder integration
  - Voice synthesis demonstration