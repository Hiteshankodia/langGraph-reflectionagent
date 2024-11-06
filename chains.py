from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#ChatPromptTemplate the text that we sent to the llm or we get from the llm 
#MessagesPlaceholder its a place holder to put you text inside the prompt. 
from langchain_google_genai import ChatGoogleGenerativeAI

#This will review the output(twitter message) its going crticise it, give suggestion to improve it. 
#This also has a messages placholder. 
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
#Generation Prompt - This will generate the tweet that are revised over and over again from the relection_prompt. 
#Hence it will revised the tweet till it gets the perfect tweet.

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatGoogleGenerativeAI( model="gemini-1.5-flash-latest")

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm