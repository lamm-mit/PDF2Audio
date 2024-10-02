""" 
At the most basic level this service is supposed to 

1) Accept a pdf url (or whatever this thing reqires)

2) Scratchpad, prompt, and user-input instructions

3) Make an api call

4) Get back a script (thats it for now)
"""


import os
import sys

import concurrent.futures as cf
import glob
import io
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal

import gradio as gr

from loguru import logger
from openai import OpenAI
from promptic import llm
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader
from tenacity import retry, retry_if_exception_type

import re

# ========== MY MODULES ========== #

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_structs.dialogue_models import Dialogue
from utils import models



def open_ai_models():
    ai_voice_models = models.STANDARD_AUDIO_MODELS
    print(ai_voice_models[1])

def google_models():
    pass

if __name__ == "__main__":
    open_ai_models()