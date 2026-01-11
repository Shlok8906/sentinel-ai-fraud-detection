from fastapi import FastAPI
import joblib
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import random
from datetime import timedelta
from bson import ObjectId

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow browser
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI model
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["fraud_system"]
transactions = db["transactions"]

FEATURES = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19",
    "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

@app.post("/check-transaction")
def check_transaction(transaction: dict):
    try:
        df = pd.DataFrame([transaction], columns=FEATURES)

        df["Amount"] = scaler.transform(df["Amount"].values.reshape(-1,1))
        df["Time"] = scaler.transform(df["Time"].values.reshape(-1,1))

        prob = model.predict_proba(df)[0][1]

        if prob > 0.8:
            risk = "HIGH"
        elif prob > 0.5:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        # Save to MongoDB
        transactions.insert_one({
            "transaction": transaction,
            "fraud_probability": float(prob),
            "risk": risk,
            "timestamp": datetime.utcnow(),
            "status": "OPEN" if risk == "HIGH" else "CLEARED"
        })

        return {
            "fraud_probability": float(prob),
            "risk": risk
        }

    except Exception as e:
        return {"error": str(e)}
@app.get("/transactions")
def get_transactions():
    data = list(transactions.find({}, {"_id": 0}))
    return data

@app.get("/fraud-cases")
def get_frauds():
    data = list(transactions.find({"risk": "HIGH"}, {"_id": 0}))
    return data
otp_sessions = db["otp_sessions"]
@app.post("/initiate-transaction")
def initiate_transaction(transaction: dict):
    df = pd.DataFrame([transaction], columns=FEATURES)

    df["Amount"] = scaler.transform(df["Amount"].values.reshape(-1,1))
    df["Time"] = scaler.transform(df["Time"].values.reshape(-1,1))

    prob = model.predict_proba(df)[0][1]

    if prob > 0.8:
        risk = "HIGH"
    elif prob > 0.5:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    if risk == "LOW":
        transactions.insert_one({
            "transaction": transaction,
            "risk": risk,
            "fraud_probability": float(prob),
            "status": "APPROVED",
            "timestamp": datetime.utcnow()
        })
        return {"status": "APPROVED", "risk": risk}

    # Generate OTP
    otp = random.randint(100000, 999999)

    txn = {
        "transaction": transaction,
        "risk": risk,
        "fraud_probability": float(prob),
        "otp": otp,
        "attempts": 0,
        "status": "OTP_PENDING",
        "timestamp": datetime.utcnow()
    }

    result = transactions.insert_one(txn)

    return {
        "status": "OTP_REQUIRED",
        "risk": risk,
        "transaction_id": str(result.inserted_id),
        "otp": otp
    }

@app.post("/verify-otp")
def verify_otp(data: dict):
    txn_id = data.get("transaction_id")
    otp = int(data.get("otp"))

    txn = transactions.find_one({"_id": ObjectId(txn_id)})

    if not txn:
        return {"status": "NOT_FOUND"}

    if txn["status"] != "OTP_PENDING":
        return {"status": txn["status"]}

    if txn["attempts"] >= 3:
        transactions.update_one({"_id": txn["_id"]}, {"$set": {"status": "BLOCKED"}})
        return {"status": "BLOCKED"}

    if txn["otp"] != otp:
        transactions.update_one(
            {"_id": txn["_id"]},
            {"$inc": {"attempts": 1}}
        )
        return {"status": "WRONG_OTP", "attempts_left": 3 - (txn["attempts"] + 1)}

    transactions.update_one(
        {"_id": txn["_id"]},
        {"$set": {"status": "APPROVED"}}
    )

    return {"status": "APPROVED"}
