import yaml

def load_users():
    with open("users.yaml", "r") as users_file:
        return yaml.safe_load(users_file)

def authenticate(username, password):
    users = load_users()
    return username in users and users[username] == password