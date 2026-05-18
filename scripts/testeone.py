import requests

def testar_conexao_tiktok():
    # URL base da API do TikTok para desenvolvedores
    url = "https://open.tiktokapis.com/v2/user/data/add/"

    
    try:
        # Realiza uma requisição GET simples
        response = requests.get(url, timeout=10)
        
        # O status 401 ou 403 é esperado aqui porque não enviamos um Token, 
        # mas confirma que o servidor está online e processando requisições.
        if response.status_code in [200, 401, 403]:
            print(f"Status Code: {response.status_code}")
            print("Sucesso: Os servidores do TikTok estão acessíveis!")
        else:
            print(f"Atenção: Recebemos o status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("Erro: Não foi possível conectar. Verifique sua internet ou DNS.")
    except requests.exceptions.Timeout:
        print("Erro: A requisição demorou demais (Timeout).")

if __name__ == "__main__":
    testar_conexao_tiktok()