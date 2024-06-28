import logging
import math
import os
import tempfile
import time
import torch
import jax.numpy as jnp
import numpy as np
from jax.experimental.compilation_cache import compilation_cache as cc
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read

from whisper_jax import FlaxWhisperPipline
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


cc.initialize_cache("./jax_cache")
##checkpoint = "openai/whisper-large-v3"
checkpoint = "openai/whisper-large-v3"
BATCH_SIZE = 16
CHUNK_LENGTH_S = 15
NUM_PROC = 32
FILE_LIMIT_MB = 1000

logger = logging.getLogger("whisper-jax-app")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

def identity(batch):
   return batch

def tqdm_generate(inputs: dict, task: str, return_timestamps: bool):
    inputs_len = inputs["array"].shape[0]
    all_chunk_start_idx = np.arange(0, inputs_len, step)
    num_samples = len(all_chunk_start_idx)
    num_batches = math.ceil(num_samples / BATCH_SIZE)

    dataloader = pipeline.preprocess_batch(inputs, chunk_length_s=CHUNK_LENGTH_S, batch_size=BATCH_SIZE)
    model_outputs = []
    start_time = time.time()
    logger.info("translating...")

    for batch in dataloader:
        model_outputs.append(pipeline.forward(batch, batch_size=BATCH_SIZE, task=task, return_timestamps=True))
    runtime = time.time() - start_time
    logger.info("done transcription")

    logger.info("post-processing...")
    post_processed = pipeline.postprocess(model_outputs, return_timestamps=True)
    text = post_processed["text"]
    if return_timestamps:
        timestamps = post_processed.get("chunks")
        timestamps = [
            f"[{format_timestamp(chunk['timestamp'][0])} -> {format_timestamp(chunk['timestamp'][1])}] {chunk['text']}"
            for chunk in timestamps
        ]
        text = "\n".join(str(feature) for feature in timestamps)
    logger.info("done post-processing")
    return text, runtime


def transcribe_chunked_audio(inputs, task, return_timestamps):
   logger.info("loading audio file...")
   if inputs is None:
       logger.warning("No audio file")
       raise ValueError("No audio file submitted! Please provide an audio file.")
   file_size_mb = os.stat(inputs).st_size / (1024 * 1024)
   if file_size_mb > FILE_LIMIT_MB:
       logger.warning("Max file size exceeded")
       raise ValueError(
           f"File size exceeds file size limit. Got file of size {file_size_mb:.2f}MB for a limit of {FILE_LIMIT_MB}MB."
       )

   with open(inputs, "rb") as f:
       inputs = f.read()

   inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)
   inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
   logger.info("done loading")
   text, runtime = tqdm_generate(inputs, task=task, return_timestamps=return_timestamps)
   return text, runtime

if __name__ == "__main__":
   torch.cuda.empty_cache()
   pipeline = FlaxWhisperPipline(checkpoint, dtype=jnp.bfloat16, batch_size=BATCH_SIZE)
   stride_length_s = CHUNK_LENGTH_S / 6
   chunk_len = round(CHUNK_LENGTH_S * pipeline.feature_extractor.sampling_rate)
   stride_left = stride_right = round(stride_length_s * pipeline.feature_extractor.sampling_rate)
   step = chunk_len - stride_left - stride_right


   logger.info("compiling forward call...")
   start = time.time()
   random_inputs = {
       "input_features": np.ones(
           (BATCH_SIZE, pipeline.model.config.num_mel_bins, 2 * pipeline.model.config.max_source_positions)
       )
   }
   random_timestamps = pipeline.forward(random_inputs, batch_size=BATCH_SIZE, return_timestamps=True)
   compile_time = time.time() - start
   logger.info(f"compiled in {compile_time}s")

   wav_file_path = "2048.wav"
   task = "translate"
   return_timestamps = False

   # segments, sample_rate = detect_voice_segments(wav_file_path)
   logger.info("detected voice segments")

   text, runtime = transcribe_chunked_audio(wav_file_path, task, return_timestamps)
   # text, runtime = transcribe_chunked_audio(segments, sample_rate, task, return_timestamps)
   print(f"Transcription:\n{text}")
   print(f"Transcription Time (s): {runtime}")
   torch.cuda.empty_cache()


