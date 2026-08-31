from src.generation.llm import create_llm


SYSTEM_PROMPT = """
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided documents."

Do not make up information.

Keep the answer clear and concise.
"""


def generate_answer(
    query: str,
    retrieved_documents,
) -> str:
    """
    Generate an answer using the retrieved documents as context.
    """

    llm = create_llm()

    context_parts = [] #We'll put our retrieved chunks into this list.

    for document in retrieved_documents:#Take each retrieved chunk one at a time.

        source = document.metadata.get("source", "Unknown")# Our chunks have metadata.like source:pdf1,so it will retrun pdf 1 else if theri is nothing mentin so instead of returnning error it will return unknown
        page = document.metadata.get("page", "Unknown")# page:1,so it will return 1 else if theri is nothing mentin so instead of returnning error it will return unknown

        context_parts.append(
            f"Source: {source}, Page: {page}\n"
            f"{document.page_content}"
        )#Add source + page + content

    context = "\n\n---\n\n".join(context_parts)#Join all the chunks

    prompt = f"""
{SYSTEM_PROMPT}

CONTEXT:
{context} #Add the retrieved context

USER QUESTION:
{query} #giving Gemini the actual question

ANSWER:
"""

    response = llm.invoke(prompt)

    content = response.content#extracting the actual content from Gemini's response

    if isinstance(content, str):#If content is already a string,return string
        return content

    if isinstance(content, list):#If content is a list
        text_parts = [] #collect the actual text here.

        for item in content:#Go through each item and checking every item returned by Gemini.
            if isinstance(item, dict) and item.get("type") == "text":#Check whether it's text
                text_parts.append(item.get("text", ""))#Extract the text

        return "".join(text_parts)#Combine the text

    return str(content)#If Gemini gives us some unexpected format that isn't a string or list, we convert it to a string rather than crashing.