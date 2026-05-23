import os
import requests

def run_entire_pipeline(image_path: str):
    # 1. Send image to local Python web service to get CNN percentages
    print("Sending image to local CNN service...")
    try:
        response = requests.post(
            "http://127.0.0.1:8000/analyze", 
            json={"image_path": os.path.abspath(image_path)}
        )
        cnn_results = response.json()
    except Exception as e:
        print(f"Could not connect to CNN service: {e}")
        return

    if cnn_results.get("status") == "error":
        print(f"Error from CNN: {cnn_results.get('message')}")
        return

    # 2. Prepare prompt containing numeric percentages for the language model
    prompt_for_llm = (
        f"I have analyzed a skin lesion image with my CNN models and obtained the following results:\n"
        f"Melanoma probability: {cnn_results['melanoma']}\n"
        f"Carcinoma probability: {cnn_results['carcinoma']}\n\n"
        f"As an intelligent specialist assistant, analyze these medical results for me in a calm, accurate, and scientific tone and provide the necessary guidance."
    )

    print("Connecting directly to Ollama Instance...")
    
    # 3. Connect directly to the local Ollama service on port 11434
    ollama_url = "http://127.0.0.1:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": "qwen2.5:7b",  # Model that is active on your system
        "prompt": prompt_for_llm,
        "stream": False
    }
    
    try:
        response = requests.post(ollama_url, json=payload, headers=headers)
        
        if response.status_code == 200:
            output_data = response.json()
            ai_response = output_data.get('response', '')
            
            print("\n=== Analysis response from the language model (Qwen 2.5) ===\n")
            print(ai_response)
        else:
            print(f"Error communicating with Ollama: status code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure the Ollama service is running in the background on your system.")

if __name__ == "__main__":
    from PIL import Image
    test_img = "test_lesion.jpg"
    Image.new('RGB', (224, 224), color='pink').save(test_img)
    
    run_entire_pipeline(test_img)
    
    if os.path.exists(test_img):
        os.remove(test_img)