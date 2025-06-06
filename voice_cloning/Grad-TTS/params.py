from model.utils import fix_len_compatibility


# data parameters
train_filelist_path = 'resources/filelists/libri-tts/train.txt'
valid_filelist_path = 'resources/filelists/libri-tts/valid.txt'
test_filelist_path = 'resources/filelists/libri-tts/test.txt'
spk_embeds_path = '../../checkpoints/precomputed_embeddings_libritts.pt'
hubert_embeds_path = '../../checkpoints/precomputed_hubert_embeddings_libritts.pt'
cmudict_path = 'resources/cmu_dictionary'
add_blank = True
n_feats = 80
n_spks = 247
spk_emb_dim = 64
n_feats = 80
n_fft = 1024
sample_rate = 24000
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000
n_units = 1000

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = 'logs/text_encoder'
test_size = 4
n_epochs = 10000
batch_size = 4
learning_rate = 2e-6
seed = 37
save_every = 1
out_size = fix_len_compatibility(2*22050//256)

chkpt = './checkpts/gradtts_44.pt'
unit_encoder_path = './checkpts/grad_1_unit.pt'
