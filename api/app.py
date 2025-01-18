from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from api.database import insert_flagged_message, get_flagged_messages, create_db
from torch.nn.functional import softmax


app = FastAPI()


model = AutoModelForSequenceClassification.from_pretrained('./toxic_comment_model')
tokenizer = AutoTokenizer.from_pretrained('./toxic_comment_model')


templates = Jinja2Templates(directory="templates")


create_db()


app.mount("/static", StaticFiles(directory="api/static"), name="static")


class ChatMessage(BaseModel):
    message: str
    user_id: str


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Toxic Comment Moderation API"}


@app.post("/moderate/")
async def moderate_message(chat_message: ChatMessage):
    try:
        
        inputs = tokenizer(chat_message.message, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits

        
        probabilities = softmax(logits, dim=-1)

        
        predicted_class = torch.argmax(probabilities, dim=-1).item()

        
        prediction = 'Toxic Comment' if predicted_class == 1 else 'Non-Toxic Comment'

        
        insert_flagged_message(chat_message.message, prediction, chat_message.user_id)
        
        return {"message": chat_message.message, "prediction": prediction}

    except Exception as e:
        return {"error": f"An error occurred during message moderation: {str(e)}"}


@app.get("/dashboard/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    
    flagged_messages = get_flagged_messages()
    
    return templates.TemplateResponse("dashboard.html", {"request": request, "messages": flagged_messages})


@app.get("/get-latest-flagged-messages")
async def get_latest_flagged_messages():
    
    flagged_messages = get_flagged_messages()
    return {"messages": flagged_messages}


@app.post("/predict")
async def predict(chat_message: ChatMessage):
    try:
        
        inputs = tokenizer(chat_message.message, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        
        probabilities = softmax(logits, dim=-1)

        
        predicted_class = torch.argmax(probabilities, dim=-1).item()

        
        prediction = 'Toxic Comment' if predicted_class == 1 else 'Non-Toxic Comment'

        return {"message": chat_message.message, "prediction": prediction}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
