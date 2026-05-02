import ollama
import requests
import json
import os

MODELO_OLLAMA = "llama3.2" 

def llamar_api(costo, prioridad):
    url = "http://127.0.0.1:8000/predict"
    try:
        res = requests.post(url, json={"costo": costo, "prioridad": prioridad.capitalize()}, timeout=10)
        return res.json()
    except Exception as e:
        return {"error": f"Error de red: {str(e)}"}

def explicar_resultado(costo, prioridad, resultado):
    prompt = f"""
    CONTEXTO: Inventario Industrial.
    DATOS: Costo {costo} USD, Prioridad {prioridad}, Stock sugerido {resultado['prediccion']}.
    TAREA: Justifica el stock en una sola frase técnica de máximo 15 palabras. 
    PROHIBIDO: Dar consejos financieros o introducciones.
    """
    response = ollama.generate(model=MODELO_OLLAMA, prompt=prompt)
    return response['response'].strip()

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("====================================================")
    print("      🌟 ASISTENTE DE INVENTARIO ACTIVO            ")
    print("====================================================")
    print("Escribe tu reporte o 'salir' para terminar.\n")

    while True:
        user_input = input("👉 SOLICITUD: ")
        if user_input.lower() in ['salir', 'exit', 'quit']: break
        if not user_input.strip(): continue

        print("⏳ Procesando...")
        try:
            res_ia = ollama.generate(model=MODELO_OLLAMA, 
                                     prompt=f'Extrae "costo" y "prioridad" de: "{user_input}". Responde solo JSON.', 
                                     format="json")
            datos = json.loads(res_ia['response'])
            res_api = llamar_api(datos['costo'], datos['prioridad'])
            
            if "error" in res_api:
                print(f"❌ ERROR: {res_api['error']}\n")
                continue

            explicacion = explicar_resultado(datos['costo'], datos['prioridad'], res_api)
            print("\n" + "─"*50)
            print(f"📊 RESULTADO: {res_api['prediccion']} unidades")
            print(f"🛡️ RANGO: [{res_api['limite_inferior']} - {res_api['limite_superior']}]")
            print(f"💡 ANÁLISIS: {explicacion}")
            print("─"*50 + "\n")
        except:
            print("⚠️ No pude entender la solicitud.")

    print("Programa finalizado.")
