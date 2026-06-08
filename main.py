from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

app = FastAPI()

# Cliente Groq
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/chat")
def chat(request: ChatRequest):
    
    

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """
                Eres SITEC, un asistente virtual de orientación en salud.
                
                REGLAS:
                - No te presentes.
                - No saludes.
                - Responde directamente a la consulta.
                - Máximo 2 frases cortas.
                - Máximo 60 palabras.
                - No diagnostiques enfermedades.
                - No indiques dosis específicas.
                - Puedes mencionar medicamentos de venta libre de forma general cuando sean comúnmente utilizados para aliviar síntomas leves.
                - Aclara que la persona debe leer las indicaciones del producto o consultar a un profesional de salud.
                - Si detectas síntomas potencialmente graves, recomienda atención médica profesional de forma breve.

                Ejemplos:
                - Para dolor leve o fiebre: puede mencionarse paracetamol.
                - Para congestión nasal leve: pueden mencionarse soluciones salinas.
                - Para irritación de garganta: pueden mencionarse pastillas para la garganta.

                No recomiendes medicamentos de prescripción.
                """
            },
            {
                "role": "user",
                "content": request.message
            }
        ],
        temperature=0.5,
        max_tokens=100
    )
    
    

    return {
        "response": completion.choices[0].message.content
    }
