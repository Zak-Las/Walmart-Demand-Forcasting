import json
import os
import sys
import pandas as pd
import traceback

# --- MONKEY PATCH: Fix AWS Lambda Multiprocessing Crash ---
import multiprocessing.queues

class SimpleQueue:
    def __init__(self, *args, **kwargs): pass
    def put(self, *args, **kwargs): pass
    def get(self, *args, **kwargs): return None
    def empty(self): return True
    def close(self): pass

multiprocessing.Queue = SimpleQueue
# -----------------------------------------------------------

from neuralforecast import NeuralForecast

# --- ENVIRONMENT CONFIG ---
os.environ["PL_ACCELERATOR"] = "cpu"
os.environ["PL_DEVICES"] = "1"
os.environ["PL_TRAINER_DEVICES"] = "1"

# --- PATHS ---
ROOT_DIR = os.environ.get("LAMBDA_TASK_ROOT", "/var/task")
MODEL_PATH = os.path.join(ROOT_DIR, "model")

print(f"Loading model from Absolute Path: {MODEL_PATH}...")
model = NeuralForecast.load(path=MODEL_PATH)

# --- CONFIG SANITIZER ---
for m in model.models:
    if hasattr(m, "hparams"):
        m.hparams["logger"] = False
        m.hparams["enable_checkpointing"] = False
        m.hparams["accelerator"] = "cpu"
        m.hparams["devices"] = 1
        m.hparams.pop("map_location", None)

print("Model loaded and sanitized.")

def handler(event, context):
    os.chdir("/tmp")
    
    try:
        # --- PARSING LOGIC ---
        if 'body' in event:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
        else:
            body = event

        if "history" not in body or "future" not in body:
            return {"statusCode": 400, "body": "Missing history or future data"}

        history_df = pd.DataFrame(body["history"])
        future_df = pd.DataFrame(body["future"])

        # Convert dates (Required for NeuralForecast)
        history_df['ds'] = pd.to_datetime(history_df['ds'])
        future_df['ds'] = pd.to_datetime(future_df['ds'])

        # --- PREDICT ---
        forecasts = model.predict(
            df=history_df, 
            futr_df=future_df
        )
        
        # --- FIX: Convert Timestamps to Strings for JSON ---
        forecasts['ds'] = forecasts['ds'].dt.strftime('%Y-%m-%d')

        return {
            "statusCode": 200,
            "body": json.dumps(forecasts.to_dict(orient="records")),
        }

    except Exception:
        return {
            "statusCode": 500, 
            "body": json.dumps({"error": traceback.format_exc()})
        }