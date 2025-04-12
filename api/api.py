# Imports 
import torchaudio
import librosa
import os
import sys
import torch
import json
import os
import shutil
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

# Adding paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './voice_cloning/Grad-TTS')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './voice_cloning/BigVGAN')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './voice_cloning/Grad-TTS/hifigan')))
sys.path.append("./voice_cloning/Grad-TTS/hifigan")
if str(Path.cwd().parent) not in sys.path:
    sys.path.append(str(Path.cwd().parent))
sys.path.append("./voice_cloning/textlesslib")

import params
from voice_cloning.speaker_encoder import ecapa_tdnn
from model import unit_tts, unit_encoder, tts
from text.symbols import symbols

from meldataset import mel_spectrogram
from hifigan.models import Generator as HiFiGAN
from env import AttrDict
from model.text_encoder import TextEncoder
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
from voice_cloning.textlesslib.textless.data.speech_encoder import SpeechEncoder



# Initialize configs
encoders_config = {
    'nsymbols': len(symbols) + 1 if params.add_blank else len(symbols),
    'n_spks': params.n_spks,
    'spk_emb_dim': params.spk_emb_dim,
    'n_enc_channels': params.n_enc_channels,
    'filter_channels': params.filter_channels,
    'filter_channels_dp': params.filter_channels_dp,
    'n_enc_layers': params.n_enc_layers,
    'enc_kernel': params.enc_kernel,
    'enc_dropout': params.enc_dropout,
    'n_heads': params.n_heads,
    'window_size': params.window_size,
    'n_units': params.n_units
}

mel_spectrogram_config = {
    'n_feats': params.n_feats,
    'n_fft': params.n_fft,
    'sample_rate': params.sample_rate,
    'hop_length': params.hop_length,
    'win_length': params.win_length,
    'f_min': params.f_min,
    'f_max': params.f_max
}

decoder_config = {
    'dec_dim': params.dec_dim,
    'beta_min': params.beta_min,
    'beta_max': params.beta_max,
    'pe_scale': params.pe_scale
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models using configs

# Speaker Encoder
print('Initializing the Speaker Encoder...')
ecapa_state_dict = torch.load('./checkpoints/speaker_encoder/speaker_encoder.pt')
ecapa = ecapa_tdnn.ECAPA_TDNN_SMALL_WTTH_PROJ()
ecapa.load_state_dict(ecapa_state_dict)
ecapa.eval().to(device)
ecapa.requires_grad = False
print('Speaker Encoder is initialized')


# Text Encoder
print('Initializing the TTS pipeline components...')
grad_tts_state_dict = torch.load('./checkpoints/grad_tts_58.pt', map_location=device)
text_encoder_state_dict = OrderedDict()

print('Initializing the text_encoder...')
for key, value in grad_tts_state_dict.items():
    if "encoder." in key:
        new_key = key.replace("encoder.", "", 1)
        text_encoder_state_dict[new_key] = value

text_encoder = TextEncoder(
    n_vocab=encoders_config['nsymbols'],
    n_feats=mel_spectrogram_config['n_feats'],
    n_channels=encoders_config['n_enc_channels'],
    filter_channels=encoders_config['filter_channels'],
    filter_channels_dp=encoders_config['filter_channels_dp'],
    n_heads=encoders_config['n_heads'],
    n_layers=encoders_config['n_enc_layers'],
    kernel_size =encoders_config['enc_kernel'],
    p_dropout=encoders_config['enc_dropout'],
    window_size=encoders_config['window_size']
).to(device)
text_encoder.load_state_dict(text_encoder_state_dict)
print('Text Encoder is initialized')

# GradTTS
tts_model = tts.GradTTS(
    n_vocab=encoders_config['nsymbols'],
    n_spks=encoders_config['n_spks'],
    spk_emb_dim=encoders_config['spk_emb_dim'],
    n_enc_channels=encoders_config['n_enc_channels'],
    filter_channels=encoders_config['filter_channels'],
    filter_channels_dp=encoders_config['filter_channels_dp'],
    n_heads=encoders_config['n_heads'],
    n_enc_layers=encoders_config['n_enc_layers'],
    enc_kernel=encoders_config['enc_kernel'],
    enc_dropout=encoders_config['enc_dropout'],
    window_size=encoders_config['window_size'],
    n_feats=mel_spectrogram_config['n_feats'],
    dec_dim=decoder_config['dec_dim'],
    beta_min=decoder_config['beta_min'],
    beta_max=decoder_config['beta_max'],
    pe_scale=decoder_config['pe_scale']
).to(device)

tts_model.load_state_dict(grad_tts_state_dict)
tts_model.encoder = text_encoder
tts_model.eval();
print('TTS pipeline is loaded and initialized')

# Unit Grad TTS
print('Initializing the UnitTTS...')
unit_gradtts = unit_tts.UnitGradTTS(
    encoders_config['n_units'], 
    encoders_config['n_spks'], 
    encoders_config['spk_emb_dim'], 
    encoders_config['n_enc_channels'],
    encoders_config['filter_channels'], 
    encoders_config['filter_channels_dp'],
    encoders_config['n_heads'], 
    encoders_config['n_enc_layers'], 
    encoders_config['enc_kernel'], 
    encoders_config['enc_dropout'], 
    encoders_config['window_size'],
    mel_spectrogram_config['n_feats'], 
    decoder_config['dec_dim'], 
    decoder_config['beta_min'], 
    decoder_config['beta_max'], 
    decoder_config['pe_scale']
).to(device)
unit_gradtts.decoder = tts_model.decoder

print('Initializing the UnitTTS encoder...')
# unit_gradtts.encoder = unit_encoder.UnitEncoder(
#     encoders_config['n_units'], 
#     mel_spectrogram_config['n_feats'], 
#     encoders_config['n_enc_channels'], 
#     encoders_config['filter_channels'],
#     encoders_config['filter_channels_dp'], 
#     encoders_config['n_heads'], 
#     encoders_config['n_enc_layers'], 
#     encoders_config['enc_kernel'],
#     encoders_config['enc_dropout'], 
#     encoders_config['window_size'], 
#     n_contentvec=0
# ).to(device)
unit_gradtts.encoder.to(device)
print('Loading the UnitTTS encoder...')
unit_encoder_state_dict = torch.load('./checkpoints/unit_encoder_141.pt', map_location=device)
# print(unit_encoder_state_dict.keys())
unit_gradtts.encoder.load_state_dict(unit_encoder_state_dict)
print('UnitTTS encoder is initialized and loaded')

print('Load UnitTTS decoder from state dict...')
# unit_decoder_state_dict = torch.load('../voice_cloning/Grad-TTS/logs/text_encoder/grad_2_unit.pt')

unit_gradtts.encoder.requires_grad = False
unit_gradtts.to(device);
print('UnitTTS is initialized and loaded')

# Unit extractor
print('Initializing the Unit Extractor encoder...')
dense_model_name = "mhubert-base-vp_en_es_fr"
quantizer_name, vocab_size = "kmeans", 1000

unit_extractor = SpeechEncoder.by_name(
    dense_model_name=dense_model_name,
    quantizer_model_name=quantizer_name,
    vocab_size=vocab_size,
    deduplicate=True,
    need_f0=False
)

_ = unit_extractor.to(device).eval()
unit_extractor.requires_grad_(False);
print('Unit Extractor is initialized')

# Hifi-GAN
print('Initializing the Hifi-GAN...')
with open('./checkpoints/hifigan-config.json') as f:
    h = AttrDict(json.load(f))
hifigan = HiFiGAN(h)
hifigan.load_state_dict(torch.load('./checkpoints/hifigan.pt', map_location=lambda loc, storage: loc)['generator'])
_ = hifigan.to(device).eval()
hifigan.remove_weight_norm()
print('Hifi-GAN is initialized')

def tts_pipeline(
    text: str,
    cmu_dict_path: str,
    encoders_config: dict,
    spk
    ):
    text_norm = text_to_sequence(text, dictionary=cmudict.CMUDict(cmu_dict_path))
    if encoders_config['add_blank']:
        text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
    text_norm = torch.IntTensor(text_norm)
    x = text_norm.unsqueeze(0)
    x_lengths = torch.LongTensor([x.shape[-1]])
    global tts_model
    global hifigan

    y_enc, y_dec, attn = tts_model(x.to(device), x_lengths.to(device), spk.to(device), torch.tensor(50).to(device))
    
    with torch.no_grad():
        audio = hifigan.forward(y_dec).cpu().squeeze().clamp(-1, 1)
    return audio

def process_unit(encoded, sampling_rate, hop_length):
    # A method that aligns units and durations (50Hz) extracted from 16kHz audio with
    # mel-spectrograms extracted from 22,050Hz audio.

    unit = encoded["units"].to(device).tolist()
    duration = encoded["durations"].to(device).tolist()

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

def exrtact_featurse_from_reference_voice(
    audio_path: str,
    device: torch.device,
    spk_enc: ecapa_tdnn.ECAPA_TDNN_SMALL_WTTH_PROJ,
    unit_extractor: SpeechEncoder,
    mel_config: dict
    ):
    wav, sr = librosa.load(audio_path)
    wav = torch.FloatTensor(wav).unsqueeze(0)
    encoded = unit_extractor(wav)
    unit, duration = process_unit(encoded, sr, 256)
    resample_fn = torchaudio.transforms.Resample(sr, 16000)
    spk = resample_fn(wav).to(device)
    spk = spk_enc(spk)
    orig_sr = sr
    mel = mel_spectrogram(wav, mel_config['n_fft'], mel_config['n_feats'], mel_config['sample_rate'], mel_config['hop_length'],
                            mel_config['win_length'], mel_config['f_min'], mel_config['f_max'], center=False).squeeze()
    
    unit, unit_lengths = unit.unsqueeze(0).to(device), torch.LongTensor([unit.shape[-1]]).to(device)
    duration = duration.unsqueeze(0).to(device)
    y, y_lengths = mel.unsqueeze(0).to(device), torch.LongTensor([mel.shape[-1]]).to(device)
    spk = spk.to(device)
    return unit, duration, spk, orig_sr, y, y_lengths, unit_lengths

def finetune_decoder(
    unit,
    duration,
    spk,
    optimizer,
    iterations,
    y,
    y_lengths,
    unit_lengths 
):
    global unit_gradtts  # Assuming 'model' is defined globally
    for iter in tqdm(range(iterations)):
        unit_gradtts.zero_grad()
        unit, unit_lengths = unit.detach(), unit_lengths.detach() 
        duration = duration.detach()
        y, y_lengths = y.detach(), y_lengths.detach() 
        spk = spk.detach()
        
        dur_loss, prior_loss, diff_loss = unit_gradtts.compute_loss(unit, unit_lengths, duration,
                                                                y, y_lengths,
                                                                spk=spk, out_size=params.out_size)
        loss = sum([dur_loss, prior_loss, diff_loss])
        loss.backward()
        optimizer.step()
    pass

# Create a directory to store uploaded audio files
AUDIO_DIR = "audio_files"
CMU_DICT_PATH = "../voice_cloning/Grad-TTS/resources/cmu_dictionary"
FINETUNE_ITERATIONS = 500
os.makedirs(AUDIO_DIR, exist_ok=True)

app = FastAPI()

@app.get("/health")
async def health_check():
    """Returns a simple health check response."""
    return JSONResponse(content={"status": "ok"})

@app.post("/clone_voice")
async def clone_voice(
    input1_type: str = Form(...),
    voice_ref_type: str = Form(...),
    input1_text: Optional[str] = Form(None),
    input1_audio: Optional[UploadFile] = File(None),
    voice_ref_audio: Optional[UploadFile] = File(None)
):
    """
    Receives voice cloning parameters, saves the reference audio,
    and returns a predefined audio file as response.
    """
    saved_ref_path = None
    try:
        # Save the uploaded voice reference audio file (optional, depending on type)
        if voice_ref_audio:
            saved_ref_path = os.path.join(AUDIO_DIR, f"reference_{voice_ref_audio.filename}")
            with open(saved_ref_path, "wb") as buffer:
                shutil.copyfileobj(voice_ref_audio.file, buffer)
            print(f"Saved reference audio to: {saved_ref_path}") # Added print for debugging
        else:
            # If voice ref type requires audio but none was provided
            raise HTTPException(status_code=400, detail="Error during saving")
        
        unit, duration, spk, orig_sr, y, y_lengths, unit_lengths = exrtact_featurse_from_reference_voice(saved_ref_path, device, ecapa, unit_extractor, mel_spectrogram_config)
        
        optimizer = torch.optim.Adam(params=unit_gradtts.decoder.parameters(), lr=params.learning_rate)
        
        finetune_decoder(unit, duration, spk, optimizer, FINETUNE_ITERATIONS, y, y_lengths, unit_lengths)
        
        generated_audio = tts_pipeline(input1_text, CMU_DICT_PATH, encoders_config, spk)
        
        torchaudio.save(f"output_{voice_ref_audio.filename}.wav", generated_audio, orig_sr)
        output_audio_path = f"output_{voice_ref_audio.filename}.wav"
        # ---------------------------------------------------

        # Check if the output file exists before returning
        if not os.path.exists(output_audio_path): 
            # You might want to create a dummy file for testing if it doesn't exist
            # with open(output_audio_path, 'w') as f:
            #     f.write("dummy content") # Or create a silent audio file
            # print(f"Warning: Output file {output_audio_path} not found. A dummy file might be needed.")
            raise HTTPException(status_code=404, detail=f"Output audio file not found at {output_audio_path}")

        # Return the generated audio file
        return FileResponse(path=output_audio_path, media_type='audio/mpeg', filename="cloned_voice.mp3")

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        # Log the error for debugging
        print(f"Error in /clone_voice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        # Ensure uploaded files are closed
        if voice_ref_audio:
            await voice_ref_audio.close()
        if input1_audio:
            await input1_audio.close()

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
