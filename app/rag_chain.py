from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
  ("system", "Ты отвечаешь строго в JSON: {format}"),
  ("user", "Ответь на вопрос на основе контекста: {question}\nКонтекст:\n{context}")
])

parser = JsonOutputParser()
chain = (prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | parser)

def rag_answer(question: str, context: str) -> dict:
    return chain.invoke({"question": question, "context": context, "format": parser.get_format_instructions()})
