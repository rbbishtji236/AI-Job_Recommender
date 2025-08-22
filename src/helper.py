import fitz
import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key

llm= ChatGroq(
    api_key=groq_api_key,model="llama-3.3-70b-versatile"
)

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from a PDF file.
    
    Args:
        uploaded_file (str): The path to the PDF file.
        
    Returns:
        str: The extracted text.
    """
    doc=fitz.open(stream=uploaded_file.read(),filetype="pdf")
    text=""
    for page in doc:
        text+=page.get_text()
    return text


def ask_groq(prompt,max_tokens=500):
    """
    Sends a prompt to the OpenAI API and returns the response.
    
    Args:
        prompt (str): The prompt to send to the OpenAI API.
        model (str): The model to use for the request.
        temperature (float): The temperature for the response.
        
    Returns:
        str: The response from the OpenAI API.
    """
    response = llm.invoke(prompt, max_tokens=max_tokens)
    return response.content