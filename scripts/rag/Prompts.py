from langchain.prompts import PromptTemplate


class FinancialExpertPrompt:
    """
    A class to generate and manage prompts for a world-renowned financial expert assistant.

    This class utilizes LangChain's PromptTemplate to format prompts with the provided
    context and user input. It ensures that responses adhere to the specified guidelines,
    including handling cases where the knowledge base does not contain relevant information.
    """

    def __init__(self):
        """
        Initializes the FinancialExpertPrompt with a predefined template.
        """
        self._template: str = """
            You are an intelligent assistant who is a world-renowned financial expert.
            Please use the content of the knowledge base to answer the question.
            Please list the data in the knowledge base and answer in detail. When all knowledge base content is IRRELEVANT 
            to the question, your answer must include the sentence "The answer you are looking for is 
            not found in the knowledge base!"
            Here is the knowledge base:
            {context}

            The ONLY Question for you to answer: {input}
        """
        self._prompt_template: PromptTemplate = PromptTemplate.from_template(self._template)

    def get_prompt_template(self) -> PromptTemplate:
        """
        Retrieves the PromptTemplate instance.

        Returns:
            PromptTemplate: The configured prompt template.
        """
        return self._prompt_template
