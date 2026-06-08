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
                - Puedes sugerir medicamentos de venta libre comúnmente utilizados para aliviar síntomas leves.
                - Puedes mencionar varias opciones cuando corresponda.
                - No recomiendes medicamentos que requieran receta médica.
                - Recomienda leer las indicaciones del producto.
                - Si hay signos de gravedad, recomienda atención médica.
                
                Ejemplos:
                - Fiebre o dolor leve: paracetamol, ibuprofeno.
                - Acidez estomacal: antiácidos.
                - Congestión nasal leve: solución salina nasal.
                - Tos leve: jarabes expectorantes o pastillas para la garganta.
                - Diarrea leve: sales de rehidratación oral.
                - Alergias leves: antihistamínicos de venta libre.

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
