import math
import torchaudio
import torch
# from audiocraft.utils.notebook import display_audio
# from transformers import AutoProcessor, MusicgenForConditionalGeneration

# processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
# model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# inputs = processor(
#     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
#     padding=True,
#     return_tensors="pt",
# )

# audio_values = model.generate(**inputs, max_new_tokens=256)

from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion

from audiocraft import train
from audiocraft.utils import export


# export.export_lm('saved_trainings/xps/635d9655/checkpoint.th', 'saved_trainings/checkpoints/635d9655_81/state_dict.bin')
# # # export.export_encodec(xp_encodec.folder / 'checkpoint.th', '/checkpoints/my_audio_lm/compression_state_dict.bin')
# export.export_pretrained_compression_model('facebook/encodec_32khz', 'saved_trainings/checkpoints/635d9655_81/compression_state_dict.bin')

USE_DIFFUSION_DECODER = False
# Using small model, better results would be obtained with `medium` or `large`.
# model = MusicGen.get_pretrained('facebook/musicgen-medium')
model = MusicGen.get_pretrained('saved_trainings/checkpoints/635d9655_81/')
if USE_DIFFUSION_DECODER:
    mbd = MultiBandDiffusion.get_mbd_musicgen()

"""Generate_continuation
# Args:
#     prompt (torch.Tensor): A batch of waveforms used for continuation.
#         Prompt should be [B, C, T], or [C, T] if only one sample is generated.
#     prompt_sample_rate (int): Sampling rate of the given audio waveforms.
#     descriptions (list of str, optional): A list of strings used as text conditioning. Defaults to None.
#     progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
"""

model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=30
)

prompt_waveform, prompt_sr = torchaudio.load("assets/fma_076439.wav")
prompt_duration = 5
prompt_waveform = prompt_waveform[..., :int(prompt_duration * prompt_sr)]
output = model.generate_continuation(prompt_waveform, prompt_sample_rate=prompt_sr, progress=True, return_tokens=True)

# print(output[0].shape) # torch.Size([1, 1, 960000])

torchaudio.save(f'sample_results/fma_076439_continuation_ckpt_81.wav', output[0].cpu().squeeze().unsqueeze(0), sample_rate=32000)
