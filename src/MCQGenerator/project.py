import os
import json
import pandas as pd
import traceback
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from .logger import logging


import PyPDF2
load_dotenv()  # take environment variables from .env.
KEY = os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(openai_api_key=KEY,model_name="gpt-3.5-turbo", temperature=0.5)

# template one for generating mcqs based on subject, tone, number and few response format examples.
TEMPLATE = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables= ["text","number","subject","tone","response_json"],
    template=TEMPLATE
)

quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation = PromptTemplate(input_variables = ["subject","quiz"],
                                 template = TEMPLATE2
                                 )

logging.info("hi, i am going to start my excution...")

chain2 = LLMChain(llm = llm,prompt = quiz_evaluation,output_key = "review",verbose = "True")
generate_evaluate_chain = SequentialChain(chains= [quiz_chain,chain2],input_variables = ["text","number","subject","tone","response_json"],output_variables = ["quiz","review"])



# quiz.to_csv("machinelearning.csv",index=False)
