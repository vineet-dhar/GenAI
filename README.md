# GenAI Enterprise Knowledge Search
Knowledge Search using GenAI

# Overview
This notebook demonstrates how Retrieval Augmented Generation (RAG) and Reasoning + Acting (ReAct) can be used to create a conversational bot, capable of assisting a customer in their search for enterprise knowledge articles.

# Scenario
You are a regular employee looking for company policues relating anmything from code of conduct to paternity, paternity policy, absence and leaves, health and safety or harassment and bullying related policies.

You enter the website and use the conversational Enterprise Search called **Searchly**!

Searchly powered by GenAI and Langchain will help you in your quest to find the most accurate information by:

Providing you the precise information
Answering questions
Providing you a summary from embeddings and indexed information
Helping you find the necessary information in a contextually conversational manner
Questions Searchly is not trained to answer get responded in a conversational manner

# Objective & Requirements
Your objective is to develop Searchly!

There is one main requirement: you will need to make sure that this chatbot is grounded. Grounding refers to the process of connecting LLMs with knowledge sources, such as databases.

In practice, this means that Searchly should leverage:

The existing catalog of policies. Searchly should not produce data that 7is not part of this catalog.
The existing product catalog of Cymbal Grocery. GroceryBot should not suggest products that are not part of this catalog.
A set of precomputed semantic excerpts catering to unique questions.
To do this, you can use an approach called Retrieval Augmented Generation (RAG), which attempts to mitigate the problem of hallucination by inserting factual information into the prompt which is sent to the LLM.


# Implementation
For demo purposes, this notebook will only use local databases. The following setup is adopted:

# Costs
This tutorial uses billable components of Google Cloud:

Vertex AI Generative AI Studio
Learn about Vertex AI pricing, and use the Pricing Calculator to generate a cost estimate based on your projected usage.

# Getting Started

**Install libraries**

    !pip install --upgrade google-cloud-aiplatform==1.27.0 langchain==0.0.196 faiss-cpu==1.7.4 --user

# Import libraries

    import glob
    import pprint
    from typing import Any, Iterator, List
    
    from langchain.agents import AgentType, initialize_agent
    from langchain.document_loaders import TextLoader
    from langchain.embeddings import VertexAIEmbeddings
    from langchain.llms import VertexAI
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import Document
    from langchain.tools import tool
    from langchain.vectorstores import FAISS
    from langchain.vectorstores.base import VectorStoreRetriever
    from tqdm import tqdm

# Initialize models

    llm = VertexAI(
        model_name="text-bison@001",
        max_output_tokens=256,
        temperature=0,
        top_p=0.8,
        top_k=40,
    )

    embedding = VertexAIEmbeddings()


# Create retrievers
As mentioned earlier, the objective is to leverage information from closed-domain databases in order to provide more context to the LLM. To do so, you will create a retriever in Langchain capable of interacting with the local vector database.

!gsutil -m cp -r "gs://genai-document-library*" .

You then define a set of functions to enable the creation of the two vector databases.

def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    """Yield successive n-sized chunks from lst.

    Args:
        lst: The list to be chunked.
        n: The size of each chunk.

    Yields:
        A list of the next n elements from lst.
    """

    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_docs_from_directory(dir_path: str) -> List[Document]:
    """Loads a series of docs from a directory.

    Args:
      dir_path: The path to the directory containing the docs.

    Returns:
      A list of the docs in the directory.
    """

    docs = []
    for file_path in glob.glob(dir_path):
        loader = TextLoader(file_path)
        docs = docs + loader.load()
    return docs


def create_retriever(top_k_results: int, dir_path: str) -> VectorStoreRetriever:
    """Create a retriever from a list of top results and a list of web pages.

    Args:
        top_k_results: number of results to return when retrieving
        dir_path: List of web pages.

    Returns:
        A retriever.
    """

    BATCH_SIZE_EMBEDDINGS = 5
    docs = load_docs_from_directory(dir_path=dir_path)
    doc_chunk = chunks(docs, BATCH_SIZE_EMBEDDINGS)
    for (index, chunk) in tqdm(enumerate(doc_chunk)):
        if index == 0:
            db = FAISS.from_documents(chunk, embedding)
        else:
            db.add_documents(chunk)

    retriever = db.as_retriever(search_kwargs={"k": top_k_results})
    return retriever

You are now ready to create the Vector DBs using the function defined in the previous step. Each Vector DB will provide a retriever instance, a Python object that, given a query, will provide a list of documents matching that query.

You will create:

chunks_retriever = create_retriever(top_k_results=2, dir_path="./chunks/*")
docs_retriever = create_retriever(top_k_results=5, dir_path="./docs/*")

Now you are ready to test the retrievers! 

docs = chunks_retriever.get_relevant_documents("Is there a parental policy?")
pprint.pprint([doc.metadata for doc in docs])

docs = docs_retriever.get_relevant_documents("Is there a maternity policy?")
pprint.pprint([doc.metadata for doc in docs])

# Agent
Now that you have created the retrievers, it's time to create the Langchain Agent, which will implement a ReAct-like approach.

You will first create the two tools to leverage the two retriever objects defined previously, chunks_retriever and docs_retriever

@tool(return_direct=True)
def retrieve_chunks(query: str) -> str:
    """
    Searches the catalog to find data for the query.
    Return the output without processing further.
    """
    docs = chunks_retriever.get_relevant_documents(query)

    return (
        f"Select the data you would like to explore further about {query}: [START CALLBACK FRONTEND] "
        + str([doc.metadata for doc in docs])
        + " [END CALLBACK FRONTEND]"
    )

@tool(return_direct=True)
def retrieve_docs(query: str) -> str:
    """Searches the catalog to find products for the query.
    """
    docs = docs_retriever.get_relevant_documents(query)
    return (
        f"I found these products about {query}:  [START CALLBACK FRONTEND] "
        + str([doc.metadata for doc in docs])
        + " [END CALLBACK FRONTEND]"
    )

You will define chunks_selector, a tool that will be used by the Agent to capture the action of the user selecting a policy. The path of chunks is used as an identifier of that policy.

@tool
def chunks_selector(path: str) -> str:
    """
    Use this when the user selects a policy.
    You will need to respond to the user telling what are the options once a policy is selected.
    """
    return "I can help you with further information - Just Ask away!"



docs = load_docs_from_directory("./chunks/*")
chunks_detail = {doc.metadata["source"]: doc.page_content for doc in docs}


@tool
def get_chunks_detail(path: str) -> str:
    """
    Use it to find more information for a specific policy, such as the policy entitlements, coverage, etc.
    """
    try:
        return chunks_detail[path]
    except KeyError:
        return "Could not find the details for this policy"


# Creating the agent

The agent will be initialised with the type CONVERSATIONAL_REACT_DESCRIPTION. To know more about it, have a look at the relative documentation and other agent types.

memory = ConversationBufferMemory(memory_key="chat_history")
memory.clear()

tool = [
    retrieve_chunks,
    retrieve_docs,
    get_chunks_detail,
    chunks_selector,
]
agent = initialize_agent(
    tool,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)


# Let's get started

agent.run("I would like to know about employee policies?")
agent.run("Can you tell me more about maternity policy?")
agent.run("How many leaves to we have as per maternity policy?")
agent.run("Can you summarise harassment policy?")
agent.run("what is the employee code of conduct?")
agent.run("For how long can I be absent from work?")
agent.run("How can employees follow health and safety policy?")
