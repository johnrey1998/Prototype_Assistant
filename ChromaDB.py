import os
import shutil

from ConstructTree import construct_tree
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Define the path to the Chroma vector store
CHROMA_PATH = "chroma"

# Initialize the OpenAIEmbeddings object
embd = OpenAIEmbeddings()


def create_vector_db():
    """
    Create and populate a Chroma vector database with document texts and their summaries.
    If a database already exists, it will be deleted and recreated.
    """
    # Delete the existing database if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Return leaf_texts and results from the construct_tree function
    leaf_texts, results = construct_tree()

    # Initialize all_texts with leaf_texts
    all_texts = leaf_texts.copy()
    # WIP all_metadata = leaf_metadata.copy()

    # Iterate through the results to extract summaries from each level and add them to all_texts
    for level in sorted(results.keys()):
        # Extract summaries from the current level's DataFrame
        summaries = results[level][1]["summaries"].tolist()

        # WIP level_metadata = [{"level": level, "cluster": idx} for idx in range(len(summaries))]

        # Extend all_texts with the summaries from the current level
        all_texts.extend(summaries)
        # WIP all_metadata.extend(level_metadata)

    # Now, use all_texts to build the vector store with Chroma
    Chroma.from_texts(texts=all_texts, embedding=embd, persist_directory=CHROMA_PATH)

if __name__ == '__main__': create_vector_db()