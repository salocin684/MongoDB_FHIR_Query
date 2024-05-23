import os

import langchain
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
import sqlalchemy
from langchain_openai import OpenAI, ChatOpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class Response(BaseModel):
    filter_documents: dict = Field(description="The MongoDB query filter document, provided as a dictionary.")
    sort_document: dict = Field(description="The MongoDb query's sort documents and makes sure only the asked fields are shown, provided as a dictionary. ordering must be 1 (for ascending) or -1 (for descending)")


PROMPT = """
I need your assistance to generate MongoDB query filter documents & query sort documents based on a user's question. 
These will then be used to query the MongoDB database using 'Pymongo'
Here's the task broken down into steps:

1. Consider the user's question provided under the prefix 'User Question:'. The question will be asking for specific \
information from a MongoDB database.

2. Use the table schema provided under the prefix 'Table Schema:'. This schema outlines the structure of the database.

3. Refer to the schema description provided under the prefix 'Schema Description:'. This description explains what each field in the schema represents.

4. Generate a the correct information for a query that retrieves the information requested in the user's question. \
The query should be compatible with the provided table schema and should consider the schema description.

5. When needed always sort the document to only return the required fields that were asked for. For example, if the user asks for the patient's first name, \
you should only return the first name and not all the fields.

Now, let's proceed with the actual task:

- User Question: "{user_message}"
- Table Schema: 
```
{table_schema}
```

- Schema Description: 
```
{schema_description}
```

Follow these instructions to format your answer.
Make sure to account for the correct datatypes.
{format_instructions}
""".strip()

TABLE_SCHEMA = {
    "fields": [
        {"name": "_id", "type": "ObjectId", "description": "Unique identifier (MongoDB specific)"},
        {"name": "Id", "type": "String", "description": "Unique identifier for the individual"},
        {"name": "BIRTHDATE", "type": "Date", "description": "Date of birth"},
        {"name": "DEATHDATE", "type": "String", "description": "Date of death (if applicable)"},
        {"name": "SSN", "type": "String", "description": "Social Security Number"},
        {"name": "DRIVERS", "type": "String", "description": "Driver's License Number"},
        {"name": "PASSPORT", "type": "String", "description": "Passport Number"},
        {"name": "PREFIX", "type": "String", "description": "Name prefix (e.g., Mr., Mrs.)"},
        {"name": "FIRST", "type": "String", "description": "First name"},
        {"name": "LAST", "type": "String", "description": "Last name"},
        {"name": "SUFFIX", "type": "String", "description": "Name suffix (if applicable)"},
        {"name": "MAIDEN", "type": "String", "description": "Maiden name (if applicable)"},
        {"name": "MARITAL", "type": "String", "description": "Marital status"},
        {"name": "RACE", "type": "String", "description": "Race"},
        {"name": "ETHNICITY", "type": "String", "description": "Ethnicity"},
        {"name": "GENDER", "type": "String", "description": "Gender"},
        {"name": "BIRTHPLACE", "type": "String", "description": "Birthplace"},
        {"name": "ADDRESS", "type": "String", "description": "Residential address"},
        {"name": "CITY", "type": "String", "description": "City"},
        {"name": "STATE", "type": "String", "description": "State"},
        {"name": "COUNTY", "type": "String", "description": "County"},
        {"name": "FIPS", "type": "Integer", "description": "Federal Information Processing Standards code"},
        {"name": "ZIP", "type": "String", "description": "ZIP code"},
        {"name": "LAT", "type": "Float", "description": "Latitude"},
        {"name": "LON", "type": "Float", "description": "Longitude"},
        {"name": "HEALTHCARE_EXPENSES", "type": "Float", "description": "Healthcare expenses"},
        {"name": "HEALTHCARE_COVERAGE", "type": "Float", "description": "Healthcare coverage"},
        {"name": "INCOME", "type": "Integer", "description": "Annual income"}
    ]
}



SCHEMA_DESCRIPTION = """
The collection you're querying on is called `try`.

Here is the description to determine what each key represents:

- `_id`: Unique identifier (MongoDB specific).
- `Id`: Unique identifier for the individual.
- `BIRTHDATE`: Date of birth.
- `DEATHDATE`: Date of death (if applicable).
- `SSN`: Social Security Number.
- `DRIVERS`: Driver's License Number.
- `PASSPORT`: Passport Number.
- `PREFIX`: Name prefix (e.g., Mr., Mrs.).
- `FIRST`: First name.
- `LAST`: Last name.
- `SUFFIX`: Name suffix (if applicable).
- `MAIDEN`: Maiden name (if applicable).
- `MARITAL`: Marital status.
- `RACE`: Race.
- `ETHNICITY`: Ethnicity.
- `GENDER`: Gender.
- `BIRTHPLACE`: Birthplace.
- `ADDRESS`: Residential address.
- `CITY`: City.
- `STATE`: State.
- `COUNTY`: County.
- `FIPS`: Federal Information Processing Standards code.
- `ZIP`: ZIP code.
- `LAT`: Latitude.
- `LON`: Longitude.
- `HEALTHCARE_EXPENSES`: Healthcare expenses.
- `HEALTHCARE_COVERAGE`: Healthcare coverage.
- `INCOME`: Annual income.
""".strip()


# OTHER VARIABLES

# METHOD
def generate_query_chain() -> Runnable:
    parser = PydanticOutputParser(pydantic_object=Response)
    llm = ChatOpenAI(
        model_name=os.getenv("MODEL_NAME"),
        temperature=int(os.getenv("DEFAULT_TEMPERATURE")),
    )

    parser = OutputFixingParser.from_llm(
        llm=llm,
        parser=parser
    )
    # https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.prompt.PromptTemplate.html
    promptTemplate = PromptTemplate(
        input_types={
            "user_message": str,
            "table_schema": dict,
            "schema_description": str,
            "format_instructions": str
        },
        partial_variables={
            "table_schema": TABLE_SCHEMA,
            "schema_description": SCHEMA_DESCRIPTION,
            "format_instructions": parser.get_format_instructions()
        },
        input_variables=["user_message"],
        output_parser=parser,
        template=PROMPT
    )

    return LLMChain(
        llm=llm,
        output_parser=parser,
        prompt=promptTemplate,
    )