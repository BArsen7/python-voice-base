from base.interface import llama, whisper

def transcribe(model_path: str, audio_path: str):
    transformer = whisper.WhisperBase(model_path)
    transformer.initialize()

    result = transformer.transcribe(audio_path)

    transformer.release()

    return result

def generate(model_path: str, prompt: str):
    transformer = llama.LlamaBase(model_path)
    transformer.initialize()

    result = transformer.generate(prompt)

    transformer.release()

    return result

