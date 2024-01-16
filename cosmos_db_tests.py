from azure.cosmos import exceptions, CosmosClient, PartitionKey
import json

endpoint = "https://cosmos-futurebrand.documents.azure.com:443/"
key = "4e89bb8afb624069bf23d504dba6508f"
database_id = "futurebrand-cosmosdb"
container_id = "futurebrand"

client = CosmosClient(endpoint, key)

database = client.get_database_client(database_id)

container = database.get_container_client(container_id)

data_to_insert = {
    "id": "1",
    "name": "exemplo",
    "description": "dados do vscode"
}

container.create_item(body=data_to_insert)

print("Data inserted successfully.")
