# From Chat GPT this code shows how to process scaned vs. digital (layerd text) pdf and what packages
# are needed to process them accordingly

from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from pathlib import Path
import pytesseract
from PIL import Image
import io

def is_scanned_pdf(pdf_path):
    """
    Checks if a PDF is likely to be a scanned document by attempting to extract text.
    Returns True if no text is found, indicating it might be a scanned document.
    """
    # Attempt to extract text using pdfminer.six
    text = extract_text(pdf_path)
    if text.strip():
        # Text was found, it's likely a digital PDF
        return False

    # If no text, check for images to determine if it might be scanned
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        if '/XObject' in page['/Resources']:
            x_objects = page['/Resources']['/XObject'].get_object()
            for obj in x_objects.values():
                if obj['/Subtype'] == '/Image':
                    # Found an image on the page
                    return True

    # No text and no images found; it might be an unusual or empty PDF
    return False

def extract_text_from_pdf(pdf_path):
    """
    Automatically detects if a PDF is scanned or digital, and extracts text accordingly.
    """
    if not Path(pdf_path).is_file():
        raise FileNotFoundError(f"No such file: '{pdf_path}'")

    if is_scanned_pdf(pdf_path):
        # Scanned PDF: Use OCR
        print(f"'{pdf_path}' appears to be a scanned document. Using OCR for text extraction.")
        return extract_text_with_ocr(pdf_path)
    else:
        # Digital PDF: Use direct text extraction
        print(f"'{pdf_path}' appears to be a digital document. Extracting text directly.")
        return extract_text(pdf_path)

def extract_text_with_ocr(pdf_path):
    """
    Extracts text from a scanned PDF using OCR (Tesseract).
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if '/XObject' in page['/Resources']:
            x_objects = page['/Resources']['/XObject'].get_object()
            for obj in x_objects.values():
                if obj['/Subtype'] == '/Image':
                    # Extract image data
                    data = obj._data
                    image = Image.open(io.BytesIO(data))
                    # Use pytesseract to extract text from the image
                    text += pytesseract.image_to_string(image)

    return text

# Usage example
pdf_files = ["document1.pdf", "document2.pdf"]  # List of PDF files to process

for pdf_file in pdf_files:
    extracted_text = extract_text_from_pdf(pdf_file)
    print(f"Extracted text from '{pdf_file}':")
    print(extracted_text)
