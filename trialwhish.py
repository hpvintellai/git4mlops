from whisper_jax import FlaxWhisperPipline

# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-large-v2")

# JIT compile the forward call - slow, but we only do once
text = pipeline("out.wav")

# used cached function thereafter - super fast!!
text = pipeline("out.wav")
from whisper_jax import FlaxWhisperPipline

# instantiate pipeline with batching
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", batch_size=16)

text = pipeline("audio.mp3", task="translate")


# transcribe and return timestamps
outputs = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)
text = outputs["text"]  # transcription
chunks = outputs["chunks"]  # transcription + timestamps


from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# instantiate pipeline with bfloat16 and enable batching
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)

# transcribe and return timestamps
outputs = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)
