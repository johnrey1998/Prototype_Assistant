import os
import shutil

from ConstructTree import construct_tree
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

CHROMA_PATH = "chroma"

embd = OpenAIEmbeddings()

def create_vector_db():
    # Delete database if doesn't exist
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    # Return leaf_texts, results from construct_tree()
    leaf_texts, results = construct_tree()
    # Initialize all_texts with leaf_texts
    all_texts = leaf_texts.copy()
    # Iterate through the results to extract summaries from each level and add them to all_texts
    for level in sorted(results.keys()):
        # Extract summaries from the current level's DataFrame
        summaries = results[level][1]["summaries"].tolist()
        # Extend all_texts with the summaries from the current level
        all_texts.extend(summaries)
    # Now, use all_texts to build the vectorstore with Chroma
    Chroma.from_texts(texts=all_texts, embedding=embd, persist_directory=CHROMA_PATH)

if __name__ == '__main__': create_vector_db()