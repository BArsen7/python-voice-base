from java import jclass

class WhisperBase():
    def __init__(
            self,
            model_path: str,
            language: str = "ru",
            threads: int = 2, 
            beam_size: int = 5,
            best_of: int = 3,
            temperature: float = 0.0,
            temperature_increment: float = 0.2,
            translate: bool = False,
            timestamps: bool = True,
            no_context: bool = True,
    ):

        self.model_path: str = model_path
        self.language: str = language
        self.threads: int = threads
        self.beam_size: int = beam_size
        self.best_of: int = best_of
        self.temperature: float = temperature
        self.temperature_increment: float = temperature_increment
        self.translate: bool = translate
        self.timestamps: bool = timestamps
        self.no_context: bool = no_context

        self.is_initialized = False
        self.bridge = None


    def initialize(self): 
        if self.is_initialized:
            return

        WhisperBridge = jclass(
            "ru.hck.voice.bridge.WhisperBridge"
        )

        self.bridge = WhisperBridge()

        self.bridge.setLanguage(self.language)
        self.bridge.setThreads(self.threads)
        self.bridge.setBeamSize(self.beam_size)
        self.bridge.setBestOf(self.best_of)
        self.bridge.setTemperature(self.temperature)
        self.bridge.setTemperatureIncrement(self.temperature_increment)
        self.bridge.setTranslate(self.translate)
        self.bridge.setTimestamps(self.timestamps)
        self.bridge.setNoContext(self.no_context)

        self.bridge.initialize(self.model_path)
        self.is_initialized = True

    def transcribe(self, audio_path):
        if not self.is_initialized:
            raise RuntimeError("Whisper is not initialized")

        return self.bridge.transcribe(audio_path)

    def release(self):
        if self.is_initialized:
            self.bridge.release()
            self.bridge = None
            self.is_initialized = False


        