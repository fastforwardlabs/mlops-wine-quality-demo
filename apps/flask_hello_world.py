from flask import Flask
import os

app = Flask(__name__)

@app.route('/hello')
def helloIndex():
    return 'Hello World from Python Flask!'

app.run(host='127.0.0.1', port= os.environ["CDSW_PUBLIC_PORT"])