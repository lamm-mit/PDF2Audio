# PDF to Audio Converter

This code can be used to convert PDFs into audio podcasts, lectures, summaries, and more. It uses OpenAI's GPT models for text generation and text-to-speech conversion.

![image](https://github.com/user-attachments/assets/ef8a5e84-d532-4e0e-b08b-fb7be2f98469)

## Features

- Upload multiple PDF files
- Choose from different instruction templates (podcast, lecture, summary, etc.)
- Customize text generation and audio models
- Select different voices for speakers

## Installation

Follow these steps to set up PDF2Audio on your local machine using Conda:

1. Clone the repository:
   ```
   git clone https://github.com/lamm-mit/PDF2Audio.git
   cd PDF2Audio
   ```

2. Install Miniconda (if you haven't already):
   - Download the installer from [Miniconda website](https://docs.conda.io/en/latest/miniconda.html)
   - Follow the installation instructions for your operating system
   - Verify the installation:
   ```
   conda --version
   ```
   
3. Create a new Conda environment:
   ```
   conda create -n pdf2audio python=3.9
   ```

4. Activate the Conda environment:
   ```
   conda activate pdf2audio
   ```

5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

6. Set up your OpenAI API key:
   Create a `.env` file in the project root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the App

To run the PDF2Audio app:

1. Ensure you're in the project directory and your Conda environment is activated:
   ```
   conda activate pdf2audio
   ```

2. Run the Python script that launches the Gradio interface:
   ```
   python app.py
   ```

3. Open your web browser and go to the URL provided in the terminal (typically `http://127.0.0.1:7860`).

4. Use the Gradio interface to upload a PDF file and convert it to audio.

## How to Use

1. Upload one or more PDF files
2. Select the desired instruction template
3. Customize the instructions if needed
4. Click "Generate Audio" to create your audio content

## Access via 🤗 Hugging Face Spaces

[lamm-mit/PDF2Audio](https://huggingface.co/spaces/lamm-mit/PDF2Audio)

## Example result

<audio controls>
  <source src="SciAgents discovery summary - example.mp3" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

## Note

This app requires an OpenAI API key to function. 

## Credits

This project was inspired by and based on the code available at [https://github.com/knowsuchagency/pdf-to-podcast](https://github.com/knowsuchagency/pdf-to-podcast) and [https://github.com/knowsuchagency/promptic](https://github.com/knowsuchagency/promptic). 

```bibtex
@article{ghafarollahi2024sciagentsautomatingscientificdiscovery,
    title={SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning}, 
    author={Alireza Ghafarollahi and Markus J. Buehler},
    year={2024},
    eprint={2409.05556},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2409.05556}, 
}
@article{buehler2024graphreasoning,
    title={Accelerating Scientific Discovery with Generative Knowledge Extraction, Graph-Based Representation, and Multimodal Intelligent Graph Reasoning},
    author={Markus J. Buehler},
    journal={Machine Learning: Science and Technology},
    year={2024},
    url={http://iopscience.iop.org/article/10.1088/2632-2153/ad7228},
}
```

