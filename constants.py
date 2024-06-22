

main_resume_template = """
You are a highly intelligent assistant capable of analyzing and discussing resumes. \
I will provide you with a resume, and I need you to help me understand its key points, strengths, and areas for improvement. \
Additionally, please answer any specific questions I might have about the resume or provide suggestions based on its content. \

Here is the resume:
{Resume_docs}

My question is: {question}

Let's start with a summary of the candidate's qualifications and experience. What stands out the most?

"""


hyde_template = """Even if you do not know the full answer, generate a one-paragraph hypothetical answer to the below question:

{question}"""


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""