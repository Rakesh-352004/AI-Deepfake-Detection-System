from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["deepfake_app"]

users = db["users"]
history = db["history"]