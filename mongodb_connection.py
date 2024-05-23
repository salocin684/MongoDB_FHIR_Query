from pymongo import MongoClient
import re
from sqlalchemy import create_engine
from llama_index.readers.mongodb import SimpleMongoReader
from sqlalchemy.orm import sessionmaker
import pandas as pd


def connect_to_db():
    client = MongoClient("localhost", 27017)
    db = client["FHIR"]
    collection = db["data"]
    return collection


def retrieve_schema():
    client = MongoClient("localhost", 27017)
    db = client["FHIR"]
    collection_names = db.list_collection_names()
    all_schemas = {}
    for collection_name in collection_names:
        collection = db[collection_name]
        documents = collection.find({})
        schema = {}
        for doc in documents:
            for key in doc:
                if key not in schema:
                    schema[key] = type(doc[key]).__name__
        all_schemas[collection_name] = schema
    return all_schemas


def parse_mongo_query(query):
    # Extract the collection name
    pattern = r"db\.(\w+)\."
    match = re.search(pattern, query)
    if match:
        collection_name = match.group(1)
    else:
        collection_name = ""

    return collection_name

def convert_to_dict(filter_str):
    filter_dict = {}
    for element in filter_str.split(","):
        if ":" in element:
            key, value = element.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith("{"):
                value = convert_to_dict(value)
            elif value.startswith("'") or value.startswith('"'):
                value = value[1:-1]
            else:
                value = convert_value(value)
            filter_dict[key] = value
    return filter_dict


def convert_value(value):
    if value.isdigit():
        return int(value)
    elif value.replace(".", "", 1).isdigit():
        return float(value)
    elif value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        return value

def execute_mongodb_query(query_str, database_name, collection_name):
    engine = create_engine(f"mongodb:///?Server=localhost;Port=27017&Database={database_name}")
    factory = sessionmaker(bind=engine)
    session = factory()


def perform_extraction(query):
    client = MongoClient("localhost", 27017)
    db = client["FHIR"]
    collection = db["try"]
    if not query.sort_document:
        cursor = collection.find(query.filter_documents)

    else:
        cursor = collection.find(query.filter_documents).sort(query.sort_document)

    return pd.DataFrame(list(cursor))



print(retrieve_schema())
