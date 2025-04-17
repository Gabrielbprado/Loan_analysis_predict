import ollama
import pandas as pd

df = pd.read_csv("../data/gold/predictions_vs_real.csv")

def ask_llm(prompt: str, df: pd.DataFrame):

    df_sample = df.head(100).to_csv(index=False)  # Amostra para não sobrecarregar a LLM
    full_prompt = f"""
Você é um assistente de análise de dados. Use a tabela a seguir para responder perguntas onde 0  = credito bom e 1 = crédito ruim .:

{df_sample}

Pergunta: {prompt}
Responda com base nos dados.
"""
    response = ollama.chat(
        model="gemma3:latest",
        messages=[{"role": "user", "content": full_prompt}]
    )
    return response['message']['content']


while True:
    user_input = input("Digite sua pergunta (ou sair): ")
    if user_input.lower() == 'sair':
        break
    resposta = ask_llm(user_input, df)
    print("Resposta do Ollama:")
    print(resposta)
