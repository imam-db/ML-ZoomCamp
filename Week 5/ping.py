from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return "Test"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)