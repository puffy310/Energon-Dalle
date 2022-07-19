from .dalle_bart_encoder import DalleBartEncoder
from .dalle_bart_decoder import DalleBartDecoder
from .vqgan_detokenizer import VQGanDetokenizer

model_class = min-dalle
backend = "nccl"

# for parallel
tp_init_size = 1
pp_init_size = 1

#for batch manager
max_batch_size = 15
max_sequence_length = 1024
repeat_round = 2
step = 8
max_wait_time = 2
