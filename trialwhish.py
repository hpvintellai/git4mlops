from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import numpy as np
import soundfile as sf
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment

# Load the audio file
audio = AudioSegment.from_file("6621.wav", format="wav")
print("Original dBFS:", audio.dBFS)

# Increase the volume of the audio by 10 dB
louder_audio = audio + 2

# Save the final output to a new file
louder_audio.export("output-1280.wav", format="wav")
print("Modified dBFS:", louder_audio.dBFS)


data, samplerate = sf.read('output-1280.wav')

# Reduce noise
y_reduced_noise = nr.reduce_noise(y=data, sr=samplerate)

# Save the reduced noise audio
sf.write('reduced_noise.wav', y_reduced_noise, samplerate)





# instantiate pipelin
pipeline = FlaxWhisperPipline("openai/whisper-large-v2",dtype = jnp.bfloat16, batch_size=32)
#pipeline = FlaxWhisperPipline("microsoft/speecht5_asr")
### transcribe and return timestamps
outputs = pipeline("reduced_noise.wav",  task="translate", return_timestamps=True)
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

