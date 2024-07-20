import json
import psycopg2


def read_json(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
        return config
    

def get_connection():
        config_path = 'Config/config.json'
        config = read_json(config_path)
        return psycopg2.connect(
            dbname=config['database']['dbname'],
            user=config['database']['user'],
            password=config['database']['password'],
            host=config['database']['host'], 
            port=config['database']['port']
        )