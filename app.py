import argparse
import flask
from flask import request, jsonify, abort

from predict import (predict_fundus_class, classify_fundus_class,
                     multi_predict_fundus_class, multi_classify_fundus_class)

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    """landing page for API

    :return: HTML for basic landing page
    :rtype: str
    """
    return '''<h1>INSIGHT ML API</h1>
                <p>A flask api implementation for INSIGHTS internal ML and DL models.   </p>'''


@app.route('/echo', methods=['POST'])
def echo():
    """returns whatever JSON was sent to it
    useful for testing if server is receiving what you think it is

    :return: received JSON data
    :rtype: JSON
    """
    return jsonify(request.json)


@app.route('/fundus/predict', methods=['POST'])
def predict_fundus():
    if not request.json or 'image' not in request.json:
        abort(400)
    else:
        # get the base64 encoded string
        file_name = request.json['filename']
        im_b64 = request.json['image']
        pred1, pred2 = predict_fundus_class(im_b64)

        return jsonify({"name":file_name,
                        "model_1_prediction": str(pred1),
                        "model_2_prediction": str(pred2)
                        })


@app.route('/fundus/predicts', methods=['POST'])
def multi_predict_fundus():
    if not request.json:
        abort(400)
    else:
        results = multi_predict_fundus_class(request.json)

        return jsonify(results)


@app.route('/fundus/classify', methods=['POST'])
def classify_fundus():
    if not request.json or 'image' not in request.json:
        abort(400)
    else:
        # get the base64 encoded string
        file_name = request.json['filename']
        im_b64 = request.json['image']
        classification = classify_fundus_class(im_b64)

        return jsonify({"file": file_name,
                        "classification": classification})


@app.route('/fundus/classifies', methods=['POST'])
def multi_classify_fundus():
    if not request.json:
        abort(400)
    else:
        # get the base64 encoded string
        #file_name = request.json['filename']
        #im_b64 = request.json['image']
        classification = multi_classify_fundus_class(request.json)

        return jsonify(classification)


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="port to access API through on host")
    args = argParser.parse_args()

    if args.port is not None:
        app.run(host="0.0.0.0", port=args.port, threaded=True)
    else:
        app.run(host="0.0.0.0", port=5000, threaded=True)
