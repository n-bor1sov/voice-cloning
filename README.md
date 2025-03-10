# Voice Cloning Project

## Description
This project aims to develop a voice cloning system that can synthesize speech in the voice of a target speaker using only a small amount of reference audio. The system leverages state-of-the-art deep learning techniques to create natural-sounding voice clones.

## Usage
1. **Speaker Encoder Training**:
   ```
   python -m voice_cloning.speaker_encoder.train
   ```
2. **Text Encoder Training**

    Before you need to download [libri-tts](https://www.openslr.org/60) dataset and place into text_encoder_dataset directory.

   voice-cloning

   ├── notebooks

   ├── text_encoder_dataset <--Here

   └── voice_cloning
    Also download and place checkpoints of decoder model from [here](https://huggingface.co/WhiteF4lcon/DecoderGradTTS/tree/main) into voice_cloning/Grad-TTS/checkpts directory.
    Then run
    ```
   python -m voice_cloning.Grad-TTS.train_text_encoder
   ```
3. **Notebooks**:
   - See the `notebooks/` directory for example implementations of different components
    - **`notebooks/speaker_encoder.ipynb` is the simple example how to test speaker encoder on a few samples**

## What Has Been Done So Far

- We performed a small research in the field of voice cloning and speech synthesis to identify the most effective approaches
- We wrote code for training ECAPA-TDNN speaker encoder on the VoxCeleb dataset. Implemented a GE2E loss function.
- We wrote code for training text-encoder for text2audio task on libri-TTS dataset.
- Created example notebooks for different parts of the pipeline:
  - Speaker encoder training and inference
  - HuBERT content encoder integration
  - Voice synthesis demonstration