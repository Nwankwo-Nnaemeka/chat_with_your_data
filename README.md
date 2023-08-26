# Chat With Your Data
- To go Through this Project I leveraged the power of <font color ="blue"> **Langchain**</font> which is an open-source library <br> designed to create applications using large language models
- I also used <font color = "purple">**Streamlit**</font> to create a user interface. Streamlit is also, an open-source library to quickly create a user interface
- I used the <font color = "green">**OPENAI**</font> apikey to access the LLM they are also so many other LLM models that are free you can <br>acccess them by just simply creating an account on [HUGGING FACE](huggingface.co)

- The dependencies needed to recreate this is on the requirement.txt file. <br>
I used Pyngrok since I used a Jupyter notebook you can also use an npx tunnel but if youre running it on a code editor you can just run streamlit run filename.py
# PROCESSES 
### 1. Load your data
Langchain has many ways to load in files from various sources from youtube to pdfs to even code files <br> you can view the documentatations for more <br>
### 2. Split your data into relevant chunks of which there are several ways to do this
langchain have provided us with relevant ways to split our data here I used TextCharacter they are others like the RecursiveCharacterSplitter <br> 
TokenTextSplitter you can choose to split it anyhow you want depends on your use case
### 3. Embed your splits
I used the openAI Embeddings. They are various embeddings out there that rank pretty hight also see [huggingface](https://huggingface.co/spaces/mteb/leaderboard)
### 4. create your vector stores
vector stores helps you store your embeddings so then can be retrieved later.
### Retrieve Document
Given a query you retrieve the most relevant piece of information from the vector store based on similarity they are other retrieval <br> 
methods you can check up on the [Langchain Documentations](https://docs.langchain.com/docs/)
  
