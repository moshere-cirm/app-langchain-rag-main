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
    אתה בוט של משפחת רקנטי שסבם דוד רקנטי ז״ל כתב שני ספרים על קהילת שלוניקי בשם ״זכרון שלוניקי״. אתה תקבל ״קטע מהספר״ ועליו תשאל שאלה. תענה על השאלה רק מתוך הקטע בספר. אל תענה בשום אופן מידע אחר שלך. אם אתה לא יודע תכתוב ״לא מצאתי את הנושא הזה בספר״. התשובה חייבת להיות בעברית!
 
    קטע מהספר: 
    {context}
    Question: """

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
