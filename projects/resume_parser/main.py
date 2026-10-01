from pydantic import BaseModel, Field, field_validator
from typing import Optional
import instructor
from openai import OpenAI
import os 
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")


client = instructor.from_openai(
    OpenAI(api_key=openai_api_key)
)


class Experience(BaseModel):
    company: str
    title: str
    start_year: int = Field(ge=1950, le=2030)
    end_year: Optional[int] = Field(default=None, ge=1950, le=2030)
    is_current: bool = Field(default=False, description="True if this is the current role")
    highlights: list[str] = Field(description="Key achievements or responsibilties")
    

class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    graduation_year: int = Field(ge=1950, le=2030)


class ResumeData(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: str = Field(description="Professional summary in 1-2 sentences")
    skills: list[str] = Field(min_length=1, description="Technical and soft skills")
    experience: list[Experience] = Field(min_length=1)
    education: list[Education]
    total_years_experience: int = Field(ge=0)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if v is not None and "@" not in v:
            raise ValueError("Invalid email format")
        return v





def parse_resume(raw_text: str) -> ResumeData:
    """Parse raw resume text into a ResumeData object."""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=ResumeData,
        max_retries=3,
        messages=[
            {
                "role":"system",
                "content":(
                    "You are a resume parsing assistant. Extract structured data "
                    "from the resume text provided. Be precise with dates and "
                    "job titles. If information is not present, use null for "
                    "optional fields. For total_years_experience, calculate the "
                    "approximate total from the work history."
                )
            },
            {
                "role":"user",
                "content": raw_text
            }
        ],
        max_completion_tokens=800,
    )
    

sample_resume = """
Jane Chen
jane.chen@email.com | (555) 123-4567 | San Francisco, CA

Senior Software Engineer with 8 years of experience in distributed systems
and cloud infrastructure.

EXPERIENCE

Senior Software Engineer, Stripe (2021 - Present)
- Led migration of payment processing pipeline to event-driven architecture
- Reduced P99 latency by 40% through caching layer redesign
- Mentored team of 4 junior engineers

Software Engineer, Dropbox (2018 - 2021)
- Built real-time file sync engine handling 50M daily operations
- Implemented conflict resolution algorithm for concurrent edits
- Owned on-call rotation for storage infrastructure

Junior Developer, Startup Inc (2016 - 2018)
- Full-stack development with Python/Django and React
- Built internal tools that saved 20 engineering hours per week

EDUCATION
BS Computer Science, UC Berkeley, 2016

SKILLS
Python, Go, Java, Kubernetes, AWS, PostgreSQL, Kafka, Redis,
Distributed Systems, System Design, Technical Leadership
"""

minimal_resume = "John Doe, Python developer, 3 years experience at Google."

def parse_resume_safe(raw_text:str)->dict:
    try:
        resume = parse_resume(sample_resume)
        return {"success":True, "data": resume.model_dump()}
    except Exception as error:
        return {
            "success":False,
            "error": str(error)
        }
        
resume = parse_resume_safe(minimal_resume)
if resume["success"]:
    print(resume["data"])
else:
    print(f"Failed: {resume['error']}")
    
    
# print(f"Name: {resume.data.name}")
# print(f"Email: {resume.email}")
# print(f"Skills: {', '.join(resume.skills[:5])}...")
# print(f"Experience: {len(resume.experience)} roles")
# print(f"Total years: {resume.total_years_experience}")

# for exp in resume.experience:
#     current = " (current)" if exp.is_current else ""
#     print(f"  - {exp.title} at {exp.company}, {exp.start_year}-{exp.end_year or 'Present'}{current}")