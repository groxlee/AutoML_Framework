from http.server import BaseHTTPRequestHandler, HTTPServer
from settings import CheckpointSettings
from trainer.conv import ConvNet
from fuel.BaseTorch import BaseTorch
from urllib.parse import urlparse, parse_qs
from flask_cors import CORS
import sys
sys.path.insert(0, 'AutoML_Framework/AutoTorch')
import ast

class RequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "OK")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'OPTIONS, POST')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = "Received data: {}".format(post_data)
        self.wfile.write(response.encode('utf-8'))
        
        print("Received data: {}".format(post_data))
        starttime = ast.literal_eval(post_data).get('time')
        epoch = int(ast.literal_eval(post_data).get('epoch'))
        config = {'train_type': 'none', 'checkpoint_settings': {'keep_checkpoints': 5, 'interval': 3000}, 'hyperparameters': {'learning_rate': 0.0003, 'momentum': 0.9, 'num_epoch': epoch}, 'tensorboard': {'loss': True, 'accuracy': True, 'learning_rate': True}, 'use_tune': False}
        cs = CheckpointSettings()
        model = ConvNet()
        BaseTorch(model, config, cs).start()



def run(server_class=HTTPServer, handler_class=RequestHandler, port=8099):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print("Starting server on port {}...".format(port))
    httpd.serve_forever()
if __name__ == '__main__':
    CORS.origins = "*"
    run()