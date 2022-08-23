from flask import Flask, jsonify, request
from predict_car_damage import predict

app = Flask(__name__)

@app.route('/car-damage', methods=['POST'])
def detect_damage():
    # if 'image' in request.args:
    #     b64_enc = str(request.args['image'])
    #     print('here: '+b64_enc)
    #
    #     return jsonify({'predicted': predict(b64_enc)})
    # else:
    #     return 'Invalid payload', 400
    if request.data is not None:
        b64_enc = request.data.decode('ascii')
        return jsonify({'predicted': predict(b64_enc)})
    else:
        return 'Invalid payload', 400

@app.errorhandler(404)
def page_not_found(e):
    return 'The resource cannot be found', 404