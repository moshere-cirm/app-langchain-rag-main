import os

from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from basic_chain import get_model
from filter import ensemble_retriever_from_docs
from local_loader import load_txt_files
from memory import create_memory_chain
from rag_chain import make_rag_chain
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


#tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictalm-7b-instruct')
#model = AutoModelForCausalLM.from_pretrained('dicta-il/dictalm-7b-instruct', trust_remote_code=True).cuda()
#model.eval()


def create_full_chain(retriever, openai_api_key=None, chat_memory=ChatMessageHistory()):
    model = get_model("ChatGPT", openai_api_key=openai_api_key)
    system_prompt = """
    You are an intelligent Hebrew chatbot with a unique purpose. Your sole objective is to provide information and answer questions exclusively related to the content of the book Zichron Saloniki, which explores the history and culture of Jews in Thessaloniki. Your responses must reflect a deep understanding of this specific text.


The book, Zichron Saloniki, delves into the vibrant Jewish community of Thessaloniki, their rich traditions, religious practices, and the significant impact they had on the city's economic, social, and cultural development. It offers a window into the past, showcasing the community's unique heritage.


Here are the guidelines to ensure your responses remain accurate and contextually relevant:


1. All answers must have direct references to specific passages or chapters in the book, Zichron Saloniki. Provide quotes or paraphrases to support your responses.
2. Refrain from extrapolating or offering interpretations that go beyond the explicit content of the book. Keep opinions separate from the factual information presented.
3. Do not incorporate any external knowledge or information from unrelated sources. Your responses should be based solely on the content of Zichron Saloniki.
4. Try to add as much as concrete details as you can. Add relevant person names, concrete dates, concrete locations and situations. Eliminate generic answers and fluff stuff
5. In case no other guidelines provide answer in 3 seperate paragraphs

Remember, your task is to demonstrate a thorough comprehension of Zichron Saloniki, providing specific examples and references to the traditions, customs, and historical insights described within its pages. 
If a question cannot be answered based on the content of the book, simply respond with, 'I don't know,' rather than venturing beyond the provided context.
Context:    {context}
answer the following question:
     """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )


    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)
    #chain = create_memory_chain(model, rag_chain, chat_memory)
    return rag_chain


# def generate_response(context, question):
#     system_prompt = """You are a helpful AI assistant for a family that their grandfather wrote a book.
#     Use the following context which is part from the book זכרון שלוניקי and the users' chat history to help the user:
#     If you don't know the answer, just say that you don't know.
#
#     Context: {context}
#     In case the human question is in Hebrew, please give your answer in Hebrew only!
#     otherwise answer in English
#     Question: {question}"""
#
#     full_prompt = system_prompt.format(context=context, question=question)
#
#     with torch.inference_mode():
#         inputs = tokenizer(full_prompt, return_tensors='pt').input_ids.to(model.device)
#         output = model.generate(
#             inputs,
#             do_sample=True,
#             top_k=50,
#             top_p=0.95,
#             temperature=0.75,
#             max_length=300,  # Increased length for a more comprehensive response
#             min_new_tokens=5
#         )
#         return tokenizer.decode(output[0], skip_special_tokens=True)


def ask_question(chain, query):
    response = chain.stream(
        {"question": query},
        config={"configurable": {"session_id": "foo"}}
    )
    return response


def main():
    load_dotenv()

    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()

    docs = load_txt_files()
    ensemble_retriever = ensemble_retriever_from_docs(docs)
    chain = create_full_chain(ensemble_retriever)

    queries = [
        "Generate a grocery list for my family meal plan for the next week(following 7 days). Prefer local, in-season ingredients."
        "Create a list of estimated calorie counts and grams of carbohydrates for each meal."
    ]

    for query in queries:
        response = ask_question(chain, query)
        console.print(Markdown(response.content))


if __name__ == '__main__':
    # this is to quiet parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
