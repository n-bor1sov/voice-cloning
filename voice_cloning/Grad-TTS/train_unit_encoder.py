import os
import sys
if 'voice-cloning' in os.getcwd().split('\\')[-1]:
    try:
        sys.path.append(os.getcwd()+'/voice_cloning/Grad-TTS')
        sys.path.append(os.getcwd()+'/voice_cloning/Grad-TTS/hifi-gan')
        os.chdir(os.getcwd()+'/voice_cloning/Grad-TTS')
    except:
        print('Problem with directory')
    print(os.getcwd())
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import params
from model.unit_tts import UnitGradTTS
from model.unit_encoder import UnitEncoder
from data import UnitMelSpeakerDataset, UnitMelSpeakerBatchCollate
from utils import plot_tensor, save_plot, AverageMeter
from text.symbols import symbols


train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
spk_embeds_path = params.spk_embeds_path
hubert_embeds_path = params.hubert_embeds_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank
n_spks = params.n_spks
spk_emb_dim = params.spk_emb_dim

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale

n_units = params.n_units

if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=log_dir)

    print('Initializing data loaders...')
    hubert_embeds = torch.load(hubert_embeds_path)
    train_dataset = UnitMelSpeakerDataset(train_filelist_path, spk_embeds_path, hubert_embeds, cmudict_path, add_blank,
                                          n_fft, n_feats, sample_rate, hop_length,
                                          win_length, f_min, f_max)
    batch_collate = UnitMelSpeakerBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=True)
    test_dataset = UnitMelSpeakerDataset(valid_filelist_path, spk_embeds_path, hubert_embeds, cmudict_path, add_blank,
                                         n_fft, n_feats, sample_rate, hop_length,
                                         win_length, f_min, f_max)

    print('Initializing model...')
    model = UnitGradTTS(nsymbols, n_spks, spk_emb_dim, n_enc_channels,
                    filter_channels, filter_channels_dp,
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()
    
    model.load_state_dict(torch.load(params.chkpt, map_location=lambda loc, storage: loc))

    model.encoder = UnitEncoder(n_units, n_feats, n_enc_channels, filter_channels,
                    filter_channels_dp, n_heads, n_enc_layers, enc_kernel,
                    enc_dropout, window_size, n_contentvec=0).cuda()
    
    pretrained_unit_state_dict = torch.load(params.unit_encoder_path, map_location=lambda loc, storage: loc)
    unit_encoder_state_dict = OrderedDict()
    for key, value in pretrained_unit_state_dict.items():
        if "encoder." in key:
            new_key = key.replace("encoder.", "", 1)
            unit_encoder_state_dict[new_key] = value
            
    model.encoder.load_state_dict(unit_encoder_state_dict)

    for param in model.decoder.parameters():
        param.requires_grad = False

    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=params.test_size)
    i = 0
    for item in test_batch:
        mel, spk = item['y'], item['spk']
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')
        i += 1
        
    print('Start training...')
    iteration = 0
    for epoch in range(1, n_epochs + 1):
        model.eval()
        print('Synthesis...')
        with torch.no_grad():
            i = 0
            for item in test_batch:
                unit = item['unit'].unsqueeze(0).cuda()
                unit_lengths = torch.LongTensor([unit.shape[1]]).cuda()
                duration = item['duration'].unsqueeze(0).cuda()
                spk = item['spk'].unsqueeze(0).cuda()

                y_enc, y_dec, attn = model(unit, unit_lengths, duration, n_timesteps=50, spk=spk)
                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                save_plot(y_enc.squeeze().cpu(),
                          f'{log_dir}/generated_enc_{i}_unit.png')
                save_plot(y_dec.squeeze().cpu(),
                          f'{log_dir}/generated_dec_{i}_unit.png')
                save_plot(attn.squeeze().cpu(),
                          f'{log_dir}/alignment_{i}_unit.png')
                i += 1

        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        rma_dur_loss = AverageMeter()
        rma_prior_loss = AverageMeter()
        rma_diff_loss = AverageMeter()
        with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()
                unit, unit_lengths = batch['unit'].cuda(), batch['unit_lengths'].cuda()
                duration = batch['duration'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                print(x.shape, x_lengths.shape)
                spk = batch['spk'].cuda()
                dur_loss, prior_loss, diff_loss = model.compute_loss(unit, unit_lengths, duration,
                                                                     y, y_lengths,
                                                                     spk=spk, out_size=out_size)
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                logger.add_scalar('training/duration_loss', dur_loss,
                                  global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss,
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss,
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)

                rma_dur_loss.update(dur_loss.item())
                rma_prior_loss.update(prior_loss.item())
                rma_diff_loss.update(diff_loss.item())

                msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {rma_dur_loss.avg}, prior_loss: {rma_prior_loss.avg}, diff_loss: {rma_diff_loss.avg}'
                progress_bar.set_description(msg)

                iteration += 1
            print(f'dur loss mean: {sum(dur_losses) / len(dur_losses)}, prior loss mean {sum(prior_losses) / len(prior_losses)}, diff loss mean {sum(diff_losses) / len(diff_losses)}')

        msg = 'Epoch %d: duration loss = %.3f ' % (epoch, rma_dur_loss.avg)
        msg += '| prior loss = %.3f ' % rma_prior_loss.avg
        msg += '| diffusion loss = %.3f\n' % rma_diff_loss.avg
        with open(f'{log_dir}/train_unit.log', 'a') as f:
            f.write(msg)

        rma_dur_loss.reset()
        rma_prior_loss.reset()
        rma_diff_loss.reset()

        if epoch % params.save_every > 0:
            continue

        ckpt = model.encoder.state_dict()
        torch.save(ckpt, f=f"{log_dir}/unit_encoder_{epoch}.pt")