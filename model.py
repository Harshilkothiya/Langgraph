from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash', temperature=0)
embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')