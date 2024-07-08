import os
import nest_asyncio
from llama_parse import LlamaParse
from dotenv import load_dotenv

# Apply nested asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables from a .env file
load_dotenv()

# Path to the directory containing the data files
DATA_PATH = "data"

# Initialize the parser with the desired result type
parser = LlamaParse(result_type="text")

def load_documents():
    """
    Load documents from the specified data path, parse them, and convert them to the LangChain format.

    Returns:
        List[LangChainDocument]: A list of parsed documents in the LangChain format.
    """
    documents = []  # Initialize an empty list to store documents
    for file in os.listdir(DATA_PATH):  # Iterate through files in the data directory
        file_path = os.path.join(DATA_PATH, file)  # Construct the full file path
        document = parser.load_data(file_path)  # Parse the document using LlamaParse
        for chunk in document:  # Iterate through the parsed chunks
            chunk = chunk.to_langchain_format()  # Convert the chunk to LangChain format
            # WIP chunk.metadata["source"] = file_path
            documents.append(chunk)  # Add the chunk to the documents list
    print(documents)  # Print the loaded documents (for debugging purposes)
    return documents  # Return the list of documents
