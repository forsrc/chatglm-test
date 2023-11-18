from flask import Flask, request, jsonify
from chat import Chat
import atexit

model_name = "HUDM/chatglm2-6b"
#model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
chatbot = Chat(model_name=model_name,
               db_name="./db/mydb.db",
               knowledge_file="./file/knowledge.txt",
               checkpoint_dir="./data/my_checkpoint/"
               )

def save_checkpoint_on_exit():
    print("Saving on service exit...")
    chatbot.close()
    print("Saving on service exit... OK")


atexit.register(save_checkpoint_on_exit)

app = Flask("chatgpt")
app.config['TIMEOUT'] = 60

@app.route('/save', methods=['GET'])
def save_checkpoint():
    chatbot.save_checkpoint()
    return jsonify(message="save checkpoint OK")

@app.route('/load', methods=['GET'])
def load_checkpoint():
    chatbot.load_checkpoint()
    return jsonify(message="Load checkpoint OK")

@app.route('/test', methods=['GET'])
def hello_world():
    return jsonify(message="Hello, World")

@app.route('/chat', methods=['POST'])
def chat_input():
    data = request.get_json()
    if 'input' in data:
        response = chatbot.input(data['input'])
        return response
    else:
        return jsonify(error="Missing 'input' parameter"), 400


if __name__ == '__main__':
    app.run(port=8080, debug=True)