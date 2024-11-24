import os

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from basic_chain import get_model
from rag_chain import make_rag_chain
from remote_loader import load_web_page
from splitter import split_documents
from vector_store import create_vector_db
from dotenv import load_dotenv


def ensemble_retriever_from_docs(docs, embeddings=None):
    # texts = docs[:50] #split_documents(docs)
    vs = create_vector_db(docs, embeddings)
    vs_docs = vs.get()
    contents = vs_docs['documents']
    metadatas = vs_docs['metadatas']

    documents = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(contents, metadatas)
    ]

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vs.as_retriever()],
        weights=[0.5, 0.5])

    return ensemble_retriever


def main():
    load_dotenv()

    problems_of_philosophy_by_russell = "https://www.gutenberg.org/ebooks/5827.html.images"
    docs = load_web_page(problems_of_philosophy_by_russell)
    ensemble_retriever = ensemble_retriever_from_docs(docs)
    model = get_model("ChatGPT")
    chain = make_rag_chain(model, ensemble_retriever) | StrOutputParser()

    result = chain.invoke("What are the key problems of philosophy according to Russell?")
    print(result)


if __name__ == "__main__":
    # this is to quite parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

