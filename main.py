"""
from chains.query_builder_chain.query_builder_chain import QueryBuilderChain
from chains.query_builder_chain.query_builder_chain import parser as queryBuilderParser

query_builder_chain = QueryBuilderChain().chain()
query_builder_chain_input = QueryBuilderChain.populate_variables({
    "question": "Get the user's who's age is less then equal to 25",
    "user_id": "any_mongo_db_id"
})
query_builder_chain_response = query_builder_chain.run(query_builder_chain_input)
query_builder_parsed_response = queryBuilderParser.parse(query_builder_chain_response).to_dict()

print("MongoDB Raw Query", query_builder_parsed_response)
"""

from mongodb_connection import perform_extraction
from chains.generate_base_chain import generate_query_chain
from mongodb_connection import retrieve_schema
import dotenv

dotenv.load_dotenv(override=True)
chain = generate_query_chain()
response = chain.invoke(
    {
        "user_message": input("What is your question?"),
    }
)

# Print the response
print(response)
print(perform_extraction(response["text"]))