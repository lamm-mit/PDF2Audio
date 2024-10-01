import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import models


def open_ai_models():
    ai_voice_models = models.STANDARD_AUDIO_MODELS
    print(ai_voice_models[1])

def google_models():
    pass

if __name__ == "__main__":
    open_ai_models()