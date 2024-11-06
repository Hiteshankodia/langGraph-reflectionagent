from typing import List, Sequence
from dotenv import load_dotenv 

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
#End is a constant that holds the underscore which is the key for the langGraph default ending node
#Node, SO when the we reach this key the langGraph stops the execution 
#MessageGraph - That state the sequence of messages. and every node will receive as an input a list of messages. 

from chains import generate_chain, reflect_chain

REFLECT = "reflect"
GENERATE = "generate"



def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})
#The generation node function will recieve input as a state which is simply a sequence of message. 
# This method will generate out generation chain, and its going to invoke it with all the state that we have already. 
# with the generate_chain.invoke{'messages' : state} we are also putting the suggestion to improve the prompt. 


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]
# The response we get back from the llm which is usually would be with the role of the AI, then we now change it to be a human message. 
# So we simply take the content, of the message and we frame it with the role of human, and we need to return it. 


# Flow of the Graph
builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

#there will be 6 time the generate and reflect node will happen 
def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()


if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
    response = graph.invoke(inputs)
    for message in response:
        print(message.content)