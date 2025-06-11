from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import get_response_from_model

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Ini homepage chatbot"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message")

    if message:
        try:
            response = get_response_from_model(message)
            return jsonify({"answer": response})
        except Exception as e:
            print("Error saat memproses pesan:", e)
            return jsonify({"answer": "Maaf, terjadi kesalahan."})
    else:
        return jsonify({"answer": "Tidak ada pesan yang diterima."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
