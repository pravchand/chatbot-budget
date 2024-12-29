from flask import Flask, request, jsonify, send_from_directory
from rag_helper import (
    extract_text_from_pdf,
    split_text,
    setup_vector_store,
    search,
    get_completion,
    BUDGET_PDF_PATH
)
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
from qdrant_client.http.exceptions import ResponseHandlingException

app = Flask(__name__, static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def initialize_services(max_retries=3, retry_delay=5):
    """Initialize services with retry logic"""
    for attempt in range(max_retries):
        try:
            text = extract_text_from_pdf(BUDGET_PDF_PATH)
            chunks = split_text(text)
            qdrant_client = setup_vector_store(chunks)
            load_dotenv()
            openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            return text, chunks, qdrant_client, openai_client
        except ResponseHandlingException as e:
            if attempt == max_retries - 1:
                raise Exception("Could not connect to Qdrant server. Please ensure Qdrant is running.") from e
            print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    raise Exception("Failed to initialize services after maximum retries")

# Initialize clients at startup with retry logic
try:
    text, chunks, qdrant_client, openai_client = initialize_services()
except Exception as e:
    print(f"Startup Error: {str(e)}")
    raise
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400

        # Get relevant chunks
        results = search(qdrant_client, question)
        references = [obj.payload["content"] for obj in results]
        context = "\n\n".join(references)
        
        # Get AI response
        response = get_completion(openai_client, question, context)
        
        return jsonify({
            'response': response,
            'context': references  
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000) 