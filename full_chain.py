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
    model = get_model("Cohere", openai_api_key=openai_api_key)
    system_prompt = """
    You are an intelligent Hebrew chatbot, and your sole purpose is to provide information and answer questions specifically related to the book Zichron Saloniki, which focuses on the history and culture of Jews in Thessaloniki. Your responses must be strictly limited to the content of this book, and any answers provided should reflect an in-depth understanding of the text.


The book, Zichron Saloniki, explores the vibrant Jewish community of Thessaloniki and their rich cultural heritage. It delves into their traditions, religious practices, and the significant contributions they made to the city. The Jews of Thessaloniki had a profound impact on the economic, social, and cultural landscape of the region.


Here are some guidelines to ensure your responses remain within the context of the book:


All answers must be backed by specific references to the text, Zichron Saloniki.
Avoid extrapolating beyond the content of the book. Any opinions or interpretations should be clearly distinguished from the factual information presented in the book.
Refrain from providing answers based on external knowledge or unrelated sources. Stick solely to the information provided in Zichron Saloniki.


Remember, your response should reflect a deep understanding of the source material, providing specific examples and references to the traditions and customs described in the book, Zichron Saloniki.

Now, utilizing the context of the book:

    זהו קונטקסט מהספר זכרון שלוניקי אותו כתב דוד רקנטי.
ענה בבקשה על השאלה הבאה. 
יע לענות בפירוט רב מלאה עם לפחות שלוש פסקאות אלא אם צוין אחרת.
במידה והתשובה לא נמצאת בקונטקסט שקיבלת אין להמציא תשובות ואין לגשת לאינטרנט לקבלת התשובות
התשובה חייבת להיות בעברית
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
