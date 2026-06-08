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

conversation_history = [
{
                "role": "system",
                "content": """
                Eres SITEC, un asistente virtual de orientación en salud.
                
                REGLAS:
- Responde de manera amable, clara y empática.
- Usa un lenguaje cercano y fácil de entender.
- No diagnostiques enfermedades.
- No indiques dosis específicas.
- Primero identifica el síntoma principal y luego sugiere únicamente opciones de venta libre relacionadas con ese síntoma.
- No recomiendes medicamentos para síntomas que el usuario no ha mencionado.
- Evita recomendar siempre los mismos medicamentos.
- Considera distintas categorías de productos de venta libre cuando sean apropiadas.
- Si existen varias alternativas razonables, menciona más de una opción.
- Recomienda leer las indicaciones del producto.
- Prioriza nombres genéricos en lugar de marcas comerciales.
- Si existen signos de gravedad, recomienda atención médica.
- Mantén respuestas breves, entre 2 y 4 oraciones.
- Evita saludos largos o presentaciones.
- Demuestra comprensión por el malestar del usuario.
- Si falta información importante, realiza una pregunta breve antes de orientar.
                
Ejemplos orientativos:

- Dolor o fiebre leve:
  paracetamol, ibuprofeno, naproxeno.

- Acidez o reflujo:
  antiácidos, carbonato de calcio, magaldrato.

- Congestión nasal:
  solución salina nasal, inhalaciones de vapor.

- Tos con flema:
  guaifenesina, jarabes expectorantes.

- Tos seca:
  pastillas para la garganta, jarabes calmantes.

- Diarrea leve:
  sales de rehidratación oral, probióticos.

- Estreñimiento ocasional:
  fibra soluble, psyllium.

- Gases o distensión abdominal:
  simeticona.

- Mareo por viaje:
  dimenhidrinato.

- Alergias leves:
  loratadina, cetirizina.

- Irritación de garganta:
  pastillas para chupar, miel (si es apropiado).

- Sequedad ocular:
  lágrimas artificiales.

- Irritación nasal:
  solución salina nasal.

                No recomiendes medicamentos de prescripción.
                """
            },
]

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/chat")
def chat(request: ChatRequest):

    # Guardar mensaje del usuario
    conversation_history.append({
        "role": "user",
        "content": request.message
    })

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_history,
        temperature=0.5,
        max_tokens=100
    )

    response_text = completion.choices[0].message.content

    # Guardar respuesta del asistente
    conversation_history.append({
        "role": "assistant",
        "content": response_text
    })

    return {
        "response": response_text
    }
