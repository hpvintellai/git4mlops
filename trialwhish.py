from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
# instantiate pipeline with batching
pipeline = FlaxWhisperPipline("openai/whisper-large-v2",dtype = jnp.bfloat16, batch_size=16)

### transcribe and return timestamps
outputs = pipeline("amplified.wav",  task="translate", return_timestamps=True)
text = outputs["text"]  # transcription
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEXT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(text)

chunks = outputs["chunks"]  # transcription + timestamps
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHUNKS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(chunks)











##from whisper_jax import FlaxWhisperPipline
##
### instantiate pipeline
##pipeline = FlaxWhisperPipline("openai/whisper-large-v2")
##
### JIT compile the forward call - slow, but we only do once
##text = pipeline("out.wav")
##
### used cached function thereafter - super fast!!
##text = pipeline("out.wav")
##print(text)




##from whisper_jax import FlaxWhisperPipline
##import jax.numpy as jnp
##
### instantiate pipeline with bfloat16 and enable batching
##pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)
##
### transcribe and return timestamps
##outputs = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)

##from gradio_client import Client
##API_URL = "http://103.93.198.18:7860/"
####API_URL ="https://huggingface.co/spaces/sanchit-gandhi/whisper-jax-spacess"
##
##client = Client(API_URL)
##
##def trnascribe_audio(audio_path, task ="transcribe",return_timestamps=False):
##    text,runtime = client.predict(
##        audio_path,
##        task,
##        return_timestamps,
##        api_name="/predict_l",
##        )
##    return text
##
##
##output = transcribe_audio("out.wav")
##
##outout_with_timestamps = transcribe_audio("out.wav",return_timestamps=True)
##
##print(output)

