from java import jclass

"""
Это базовый класс для общения с whisper-ом в нашей парадигме
По сути это мостик между практически такой же реализацией в Java
Я искренне прошу, оставьте это так, иначе поломается примерно все, потому что оно в тупую захардкожено с двух сторон
и если поменять что то тут и не сказать мне -- просто ничего не запустится так как классы не синхронизированы, это работает точно, предлагаю не трогать
(Импорт так и должен гореть желтым, все ок)

Атрибуты класса настраиваемы, но на дефолтных значениях тоже норм фурычит, эксперименты это ваше дело -- пожалуйста
Если по приколу решить что язык это сильно число ничего страшного не произойдет, просто воткнутся дефолтные значения

Функция initialize как ни странно инициализирует обьект с заданными параметрами, можно задать их один раз на весь пайплайн и дальше в целом просто перегружать модельку раз за разом

Функция release освобождает процесс из памяти телефона, но и подгрузка каждый раз довольно ресурсоемкая штука, надо помнить

Путь к файлу и модели всегда отдает Java, так проще всем, ибо я сам хер че понимаю в том как андроид апка читает директории(работает -- не трогай)
"""

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


        