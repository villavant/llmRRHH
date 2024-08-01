
from Models import get_HF_embeddings, cosine, get_doc2vec_embeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os

def compare(resume_texts, JD_text, flag='HuggingFace-BERT'):
    JD_embeddings = None
    resume_embeddings = []
    summary = []

    if flag == 'HuggingFace-BERT':
        summary_chain = load_summary_chain()
        if JD_text is not None:
            JD_embeddings = get_HF_embeddings(JD_text)
        for resume_text in resume_texts:
            resume_embeddings.append(get_HF_embeddings(resume_text))
            summary.append(get_summary(summary_chain, resume_text))

        if JD_embeddings is not None and resume_embeddings is not None:
            cos_scores = cosine(resume_embeddings, JD_embeddings)
            return cos_scores, summary

    # Add logic for other flags like 'Doc2Vec' if necessary
    else:
        # Handle other cases
        pass

def get_summary(summary_chain, text):
    summary = summary_chain.run({"text": text})
    return summary

def load_summary_chain():
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text:\n\n{text}\n\nSummary:"
    )

    llm = ChatOpenAI(api_key=os.environ.get('OPEN_API_KEY'),
        model="gpt-3.5-turbo", temperature=1)
    summary_chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )
    return summary_chain