
import os

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))





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


resume_prompt_template="""
Here is a resume:
{resume}
    
Extract the key skills and experiences mentioned in the resume.
"""


job_desc_template = """
Here is a job description:
{job_description}

Extract the key skills and requirements mentioned in the job description.
"""

compare_prompt_template = """
Compare the following key skills and experiences extracted from a resume with the key skills and requirements from a job description:

Resume Skills and Experiences:
{resume_skills}

Job Description Skills and Requirements:
{job_skills}

Provide a detailed comparison and mention any gaps or strong matches.
"""