from java import jclass

class LlamaBase():
    def __init__(
            self,
            model_path: str,
            context_size: int = 2048,
            threads: int = 4,
            batch_size: int = 256,
            micro_batch_size: int = 128,
            max_tokens: int = 256,
            temperature: float = 0.7,
            topK: int = 20,
            topP: float = 0.8,
            system_prompt: str = "",
    ):
        self.bridge = None
        self.is_initialized = False
        
        self.model_path: str = model_path
        self.context_size: int = context_size
        self.threads: int = threads
        self.batch_size: int = batch_size
        self.micro_batch_size: int = micro_batch_size
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature
        self.topK: int = topK
        self.topP: float = topP
        self.system_prompt: str = system_prompt

    def initialize(self):
        if self.is_initialized:
            return
        
        LlamaBridge = jclass(
            "ru.hck.voice.bridge.LlamaBridge"
        )

        self.bridge = LlamaBridge()

        self.bridge.setContextSize(self.context_size)
        self.bridge.setThreads(self.threads)

        self.bridge.setBatchSize(self.batch_size)
        self.bridge.setMicroBatchSize(self.micro_batch_size)

        self.bridge.setMaxTokens(self.max_tokens)
        self.bridge.setTemperature(self.temperature)
        self.bridge.setTopK(self.topK)
        self.bridge.setTopP(self.topP)

        self.bridge.setSystemPrompt(self.system_prompt)

        self.bridge.initialize(self.model_path)
        self.is_initialized = True

    def generate(self, prompt: str) -> str:
        if not self.initialize:
            raise RuntimeError("Llama is not initialized")

        if prompt == "":
            return ""

        return self.bridge.generate(prompt)

    def release(self):
            if self.is_initialized:
                self.bridge.release()
                self.bridge = None
                self.is_initialized = False