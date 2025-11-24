import pyodbc
from keys import db_conn_str

# REGISTERING NEW DATA
def register_store(store_dict):
    # Check if store already exists (for now, check for a perfect match)
    query = """
    SELECT store_id FROM Stores 
    WHERE name = ? AND address = ? AND city = ? AND state = ?
    """
    params = (store_dict["name"], store_dict["address"], store_dict["city"], store_dict["state"])
    result = fetch_query(query, params)

    if result:
        print(f"Store {store_dict['name']} already exists.")
        return result[0][0] # Store already exists, return store_id
    
    else:
        # Insert new store and get its store_id
        store_id = create_record("Stores", store_dict, id_column="store_id")
        print(f"Registered store {store_dict['name']} with id {store_id}.")
        return store_id

# TODO: Check if user exists already
def register_user(user_dict, store_id):
    user_dict["store_id"] = store_id
    print(user_dict)
    user_id = create_record("Users", user_dict, id_column="user_id")
    print(f"Registered user {user_dict['name']} with id {user_id}.")


# GENERAL FUNCTIONS
def create_record(table_name, data, id_column):
    """Inserts a record into the specified table, returns its id.

    Parameters:
    - table_name (str): The name of the table in the database.
    - data (dict): A dictionary where keys are column names and values are the values to insert.
    """
    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?' for _ in data])
    values = tuple(data.values())

    query = f"""
    INSERT INTO {table_name} ({columns}) 
    OUTPUT INSERTED.{id_column}
    VALUES ({placeholders})
    """

    with pyodbc.connect(db_conn_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, values)
            inserted_id = cursor.fetchone()[0]
            conn.commit()

    print(f"Inserted new record into {table_name}")
    return inserted_id
    

def modify_query(query, params=None):
    """Executes a query that modifies data (INSERT, UPDATE, DELETE, TRUNCATE)."""
    conn = pyodbc.connect(db_conn_str)
    cursor = conn.cursor()
    cursor.execute(query, params or ())
    conn.commit()
    cursor.close()
    conn.close()

    

def fetch_query(query, params=None):
    """Executes a query and returns all results as a list of tuples."""
    conn = pyodbc.connect(db_conn_str)
    cursor = conn.cursor()
    cursor.execute(query, params or ())
    rows = cursor.fetchall()  # Fetch all rows
    cursor.close()
    conn.close()
    return rows


def update_value(table_name, column_name, new_value, condition_column, condition_value):
    """Updates a value in a specific column of a table based on a condition.

    Parameters:
    - table_name (str): The name of the table.
    - column_name (str): The column to update.
    - new_value: The new value to set.
    - condition_column (str): The column used for the WHERE condition.
    - condition_value: The value for the WHERE condition.
    """
    query = f"UPDATE {table_name} SET {column_name} = ? WHERE {condition_column} = ?"
    params = (new_value, condition_value)

    modify_query(query, params)
    print(f"Updated `{column_name}` in `{table_name}` where `{condition_column}` = {condition_value}")