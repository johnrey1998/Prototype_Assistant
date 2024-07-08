import os
import nest_asyncio
nest_asyncio.apply()
from llama_parse import LlamaParse
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data"
parser = LlamaParse(
    result_type="text"
)


def load_documents():
    documents = []
    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)
        document = parser.load_data(file_path)
        for chunk in document:
            chunk = chunk.to_langchain_format()
            # chunk.metadata["source"] = file_path
            documents.append(chunk)
    print(documents)
    return documents

