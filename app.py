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

def read_readme():
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, "r") as file:
            content = file.read()
            # Use regex to remove metadata enclosed in -- ... --
            content = re.sub(r'--.*?--', '', content, flags=re.DOTALL)
            return content
    else:
        return "README.md not found. Please check the repository for more information."
        
# Define multiple sets of instruction templates
INSTRUCTION_TEMPLATES = {
################# PODCAST ##################
    "podcast": {
        "intro": """Your task is to take the input text provided and turn it into an lively, engaging, informative podcast dialogue, in the style of NPR. The input text may be messy or unstructured, as it could come from a variety of sources like PDFs or web pages. 

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that could be discussed in a podcast. 

Define all terms used carefully for a broad audience of listeners.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for a high quality presentation.",
        "scratch_pad": """Brainstorm creative ways to discuss the main topics and key points you identified in the input text. Consider using analogies, examples, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your podcast should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.

Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the podcast dialogue here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it fun and exciting. 
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a very long, engaging, informative podcast dialogue here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to a general audience. 

Never use made-up names for the hosts and guests, but make it an engaging and immersive experience for listeners. Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

Make the dialogue as long and detailed as possible, while still staying on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest podcast episode you can, while still communicating the key information from the input text in an entertaining way.

At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and takeaways from their discussion. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas one last time before signing off. 

The podcast should have around 20000 words.
""",
    },
################# MATERIAL DISCOVERY SUMMARY ##################
    "SciAgents material discovery summary": {
        "intro": """Your task is to take the input text provided and turn it into a lively, engaging conversation between a professor and a student in a panel discussion that describes a new material. The professor acts like Richard Feynman, but you never mention the name.

The input text is the result of a design developed by SciAgents, an AI tool for scientific discovery that has come up with a detailed materials design.

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that could be discussed in a podcast.

Define all terms used carefully for a broad audience of listeners.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for a high quality presentation.",
        "scratch_pad": """Brainstorm creative ways to discuss the main topics and key points you identified in the material design summary, especially paying attention to design features developed by SciAgents. Consider using analogies, examples, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your description should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.

Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the podcast dialogue here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it fun and exciting. You never refer to the podcast, you just discuss the discovery and you focus on the new material design only.
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a very long, engaging, informative dialogue here, based on the key points and creative ideas you came up with during the brainstorming session. The presentation must focus on the novel aspects of the material design, behavior, and all related aspects.

Use a conversational tone and include any necessary context or explanations to make the content accessible to a general audience, but make it detailed, logical, and technical so that it has all necessary aspects for listeners to understand the material and its unexpected properties.

Remember, this describes a design developed by SciAgents, and this must be explicitly stated for the listeners.

Never use made-up names for the hosts and guests, but make it an engaging and immersive experience for listeners. Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

Make the dialogue as long and detailed as possible with great scientific depth, while still staying on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest podcast episode you can, while still communicating the key information from the input text in an entertaining way.

At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and takeaways from their discussion. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas one last time before signing off.

The conversation should have around 20000 words.
"""
    },
################# LECTURE ##################
    "lecture": {
        "intro": """You are Professor Richard Feynman. Your task is to develop a script for a lecture. You never mention your name.

The material covered in the lecture is based on the provided text. 

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be covered in the lecture. 

Define all terms used carefully for a broad audience of students.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for a high quality presentation.",
        "scratch_pad": """
Brainstorm creative ways to discuss the main topics and key points you identified in the input text. Consider using analogies, examples, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your lecture should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.

Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the lecture here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it fun and exciting. 
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a very long, engaging, informative script here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to the students.

Include clear definitions and terms, and examples. 

Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

There is only one speaker, you, the professor. Stay on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest lecture you can, while still communicating the key information from the input text in an engaging way.

At the end of the lecture, naturally summarize the main insights and takeaways from the lecture. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. 

Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas covered in this lecture one last time before class is over. 

The lecture should have around 20000 words.
""",
    },
################# SUMMARY ##################
        "summary": {
        "intro": """Your task is to develop a summary of a paper. You never mention your name.

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be summarized.

Define all terms used carefully for a broad audience.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and key facts. Think about how you could present this information in an accurate summary.",
        "scratch_pad": """Brainstorm creative ways to present the main topics and key points you identified in the input text. Consider using analogies, examples, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your summary should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms. Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the summary here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it engaging and exciting. 
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it is time to write the actual summary. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a a script here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to the the audience.

Start your script by stating that this is a summary, referencing the title or headings in the input text. If the input text has no title, come up with a succinct summary of what is covered to open.

Include clear definitions and terms, and examples, of all key issues. 

Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

There is only one speaker, you. Stay on topic and maintaining an engaging flow. 

Naturally summarize the main insights and takeaways from the summary. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. 

The summary should have around 1024 words.
""",
    },
################# SHORT SUMMARY ##################
        "short summary": {
        "intro": """Your task is to develop a summary of a paper. You never mention your name.

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be summarized.

Define all terms used carefully for a broad audience.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and key facts. Think about how you could present this information in an accurate summary.",
        "scratch_pad": """Brainstorm creative ways to present the main topics and key points you identified in the input text. Consider using analogies, examples, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your summary should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms. Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the summary here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it engaging and exciting. 
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it is time to write the actual summary. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a a script here, based on the key points and creative ideas you came up with during the brainstorming session. Keep it concise, and use a conversational tone and include any necessary context or explanations to make the content accessible to the the audience.

Start your script by stating that this is a summary, referencing the title or headings in the input text. If the input text has no title, come up with a succinct summary of what is covered to open.

Include clear definitions and terms, and examples, of all key issues. 

Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

There is only one speaker, you. Stay on topic and maintaining an engaging flow. 

Naturally summarize the main insights and takeaways from the short summary. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. 

The summary should have around 256 words.
""",
    },
}

# Function to update instruction fields based on template selection
def update_instructions(template):
    return (
        INSTRUCTION_TEMPLATES[template]["intro"],
        INSTRUCTION_TEMPLATES[template]["text_instructions"],
        INSTRUCTION_TEMPLATES[template]["scratch_pad"],
        INSTRUCTION_TEMPLATES[template]["prelude"],
        INSTRUCTION_TEMPLATES[template]["dialog"]
           )

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

# Define standard values
STANDARD_TEXT_MODELS = [
    "o1-preview-2024-09-12",
    "o1-preview",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
    "o1-mini-2024-09-12",
    "o1-mini",
    "chatgpt-4o-latest",
    "gpt-4-turbo",
    "openai/custom_model",
]

STANDARD_AUDIO_MODELS = [
    "tts-1",
    "tts-1-hd",
]

STANDARD_VOICES = [
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
]

class DialogueItem(BaseModel):
    text: str
    speaker: Literal["speaker-1", "speaker-2"]

class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]

def get_mp3(text: str, voice: str, audio_model: str, api_key: str = None) -> bytes:
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )

    with client.audio.speech.with_streaming_response.create(
        model=audio_model,
        voice=voice,
        input=text,
    ) as response:
        with io.BytesIO() as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
            return file.getvalue()


from functools import wraps

def conditional_llm(model, api_base=None, api_key=None):
    """
    Conditionally apply the @llm decorator based on the api_base parameter.
    If api_base is provided, it applies the @llm decorator with api_base.
    Otherwise, it applies the @llm decorator without api_base.
    """
    def decorator(func):
        if api_base:
            return llm(model=model, api_base=api_base)(func)
        else:
            return llm(model=model, api_key=api_key)(func)
    return decorator

def generate_audio(
    files: list,
    openai_api_key: str = None,
    text_model: str = "o1-preview-2024-09-12",
    audio_model: str = "tts-1",
    speaker_1_voice: str = "alloy",
    speaker_2_voice: str = "echo",
    api_base: str = None,
    intro_instructions: str = '',
    text_instructions: str = '',
    scratch_pad_instructions: str = '',
    prelude_dialog: str = '',
    podcast_dialog_instructions: str = '',
) -> bytes:
    # Validate API Key
    if not os.getenv("OPENAI_API_KEY") and not openai_api_key:
        raise gr.Error("OpenAI API key is required")

    combined_text = ""

    # Loop through each uploaded file and extract text
    for file in files:
        with Path(file).open("rb") as f:
            reader = PdfReader(f)
            text = "\n\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            combined_text += text + "\n\n"  # Add separation between different files' texts

    # Configure the LLM based on selected model and api_base
    @retry(retry=retry_if_exception_type(ValidationError))
    @conditional_llm(model=text_model, api_base=api_base, api_key=openai_api_key)
    def generate_dialogue(text: str, intro_instructions: str, text_instructions: str, scratch_pad_instructions: str, 
                          prelude_dialog: str, podcast_dialog_instructions: str,
                          ) -> Dialogue:
        """
        {intro_instructions}
        
        Here is the input text you will be working with:
        
        <input_text>
        {text}
        </input_text>

        {text_instructions}
        
        <scratchpad>
        {scratch_pad_instructions}
        </scratchpad>
        
        {prelude_dialog}
        
        <podcast_dialogue>
        {podcast_dialog_instructions}
        </podcast_dialogue>
        """

    # Generate the dialogue using the LLM
    llm_output = generate_dialogue(
        combined_text,
        intro_instructions=intro_instructions,
        text_instructions=text_instructions,
        scratch_pad_instructions=scratch_pad_instructions,
        prelude_dialog=prelude_dialog,
        podcast_dialog_instructions=podcast_dialog_instructions
    )

    audio = b""
    transcript = ""

    characters = 0

    with cf.ThreadPoolExecutor() as executor:
        futures = []
        for line in llm_output.dialogue:
            transcript_line = f"{line.speaker}: {line.text}"
            voice = speaker_1_voice if line.speaker == "speaker-1" else speaker_2_voice
            future = executor.submit(get_mp3, line.text, voice, audio_model, openai_api_key)
            futures.append((future, transcript_line))
            characters += len(line.text)

        for future, transcript_line in futures:
            audio_chunk = future.result()
            audio += audio_chunk
            transcript += transcript_line + "\n\n"

    logger.info(f"Generated {characters} characters of audio")

    temporary_directory = "./gradio_cached_examples/tmp/"
    os.makedirs(temporary_directory, exist_ok=True)

    # Use a temporary file -- Gradio's audio component doesn't work with raw bytes in Safari
    temporary_file = NamedTemporaryFile(
        dir=temporary_directory,
        delete=False,
        suffix=".mp3",
    )
    temporary_file.write(audio)
    temporary_file.close()

    # Delete any files in the temp directory that end with .mp3 and are over a day old
    for file in glob.glob(f"{temporary_directory}*.mp3"):
        if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 24 * 60 * 60:
            os.remove(file)

    return temporary_file.name, transcript

def validate_and_generate_audio(*args):
    files = args[0]
    if not files:
        return None, None, "Please upload at least one PDF file before generating audio."
    audio_file, transcript = generate_audio(*args)
    return audio_file, transcript, None

with gr.Blocks(title="PDF to Audio", css="""
    #header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px;
        background-color: transparent;
        border-bottom: 1px solid #ddd;
    }
    #title {
        font-size: 24px;
        margin: 0;
    }
    #logo_container {
        width: 200px;
        height: 200px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    #logo_image {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    #main_container {
        margin-top: 20px;
    }
""") as demo:
    
    with gr.Row(elem_id="header"):
        with gr.Column(scale=4):
            gr.Markdown("# Convert PDFs into an audio podcast, lecture, summary and others\n\nFirst, upload one or more PDFs, select options, then push Generate Audio.\n\nYou can also select a variety of custom option and direct the way the result is generated.", elem_id="title")
        with gr.Column(scale=1):
            gr.HTML('''
                <div id="logo_container">
                    <img src="https://huggingface.co/spaces/lamm-mit/PDF2Audio/resolve/main/logo.png" id="logo_image" alt="Logo">
                </div>
            ''')
    #gr.Markdown("")    
    submit_btn = gr.Button("Generate Audio", elem_id="submit_btn")

    with gr.Row(elem_id="main_container"):
        with gr.Column(scale=2):
            files = gr.Files(label="PDFs", file_types=["pdf"], )
            
            openai_api_key = gr.Textbox(
                label="OpenAI API Key",
                visible=True,  # Always show the API key field
                placeholder="Enter your OpenAI API Key here...",
                type="password"  # Hide the API key input
            )
            text_model = gr.Dropdown(
                label="Text Generation Model",
                choices=STANDARD_TEXT_MODELS,
                value="o1-preview-2024-09-12", #"gpt-4o-mini",
                info="Select the model to generate the dialogue text.",
            )
            audio_model = gr.Dropdown(
                label="Audio Generation Model",
                choices=STANDARD_AUDIO_MODELS,
                value="tts-1",
                info="Select the model to generate the audio.",
            )
            speaker_1_voice = gr.Dropdown(
                label="Speaker 1 Voice",
                choices=STANDARD_VOICES,
                value="alloy",
                info="Select the voice for Speaker 1.",
            )
            speaker_2_voice = gr.Dropdown(
                label="Speaker 2 Voice",
                choices=STANDARD_VOICES,
                value="echo",
                info="Select the voice for Speaker 2.",
            )
            api_base = gr.Textbox(
                label="Custom API Base",
                placeholder="Enter custom API base URL if using a custom/local model...",
                info="If you are using a custom or local model, provide the API base URL here, e.g.: http://localhost:8080/v1 for llama.cpp REST server.",
            )

        with gr.Column(scale=3):
            template_dropdown = gr.Dropdown(
                label="Instruction Template",
                choices=list(INSTRUCTION_TEMPLATES.keys()),
                value="podcast",
                info="Select the instruction template to use. You can also edit any of the fields for more tailored results.",
            )
            intro_instructions = gr.Textbox(
                label="Intro Instructions",
                lines=10,
                value=INSTRUCTION_TEMPLATES["podcast"]["intro"],
                info="Provide the introductory instructions for generating the dialogue.",
            )
            text_instructions = gr.Textbox(
                label="Standard Text Analysis Instructions",
                lines=10,
                placeholder="Enter text analysis instructions...",
                value=INSTRUCTION_TEMPLATES["podcast"]["text_instructions"],
                info="Provide the instructions for analyzing the raw data and text.",
            )
            scratch_pad_instructions = gr.Textbox(
                label="Scratch Pad Instructions",
                lines=15,
                value=INSTRUCTION_TEMPLATES["podcast"]["scratch_pad"],
                info="Provide the scratch pad instructions for brainstorming presentation/dialogue content.",
            )
            prelude_dialog = gr.Textbox(
                label="Prelude Dialog",
                lines=5,
                value=INSTRUCTION_TEMPLATES["podcast"]["prelude"],
                info="Provide the prelude instructions before the presentation/dialogue is developed.",
            )
            podcast_dialog_instructions = gr.Textbox(
                label="Podcast Dialog Instructions",
                lines=20,
                value=INSTRUCTION_TEMPLATES["podcast"]["dialog"],
                info="Provide the instructions for generating the presentation or podcast dialogue.",
            )
    
    audio_output = gr.Audio(label="Audio", format="mp3")
    transcript_output = gr.Textbox(label="Transcript", lines=20, show_copy_button=True)
    error_output = gr.Textbox(visible=False)  # Hidden textbox to store error message

    # Update instruction fields when template is changed
    template_dropdown.change(
        fn=update_instructions,
        inputs=[template_dropdown],
        outputs=[intro_instructions, text_instructions, scratch_pad_instructions, prelude_dialog, podcast_dialog_instructions]
    )

    submit_btn.click(
        fn=validate_and_generate_audio,
        inputs=[
            files, openai_api_key, text_model, audio_model, 
            speaker_1_voice, speaker_2_voice, api_base,
            intro_instructions, text_instructions, scratch_pad_instructions, 
            prelude_dialog, podcast_dialog_instructions
        ],
        outputs=[
            audio_output, 
            transcript_output,
            error_output
        ]
        ).then(
            fn=lambda error: gr.Warning(error) if error else None,
            inputs=[error_output],
            outputs=[]
        )

    # Add README content at the bottom
    gr.Markdown("---")  # Horizontal line to separate the interface from README
    gr.Markdown(read_readme())
    
# Enable queueing for better performance
demo.queue(max_size=20, default_concurrency_limit=32)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()