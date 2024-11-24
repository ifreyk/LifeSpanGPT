from dotenv import load_dotenv

from langchain.chains import RetrievalQA


class QA:
    def __init__(self, compression_retriever, llm, parser, prompt):
        self.compression_retriever = compression_retriever
        self.llm = llm
        self.prompt = prompt
        self.parser = parser

    def run_qa(self,query):
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.compression_retriever,
            chain_type_kwargs={
                "prompt": self.prompt,
                "document_variable_name": "context",
            },
        )
        answer = qa.run(query)
        result = self.parser.invoke(answer)
        return result