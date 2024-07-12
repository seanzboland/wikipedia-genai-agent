
# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['GOOGLE_API_KEY'] = apikey

# App framework
st.title('Wikipedia + GPT')
st.subheader('Create a text answer searching Wikipedia')
prompt = st.text_input('Enter Your Prompt') 

script_template = PromptTemplate(
    input_variables = ['wikipedia_research'], 
    template='Give me the answer in 1 or 2 sentences while searching this wikipedia reserch:{wikipedia_research} '
)

# Memory 
script_memory = ConversationBufferMemory(memory_key='chat_history')

# Llms
llm = GooglePalm(temperature=0.4)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(wikipedia_research=wiki_research)

    st.write(script) 
