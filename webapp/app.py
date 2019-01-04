APP_NAME = 'v1.0.0 REST API'

APP_DESCRIPTION = 'face expression with  REST API.'


import time
from flask import Flask, jsonify, g, request, redirect
from flask_swagger import swagger

app = Flask(__name__)


@app.route('/')
def root():
    return redirect('./static/swagger-ui-build/index.html')


@app.route('/spec')
def spec():
    swag = swagger(app)
    swag['info']['version'] = "multiple"
    swag['info']['title'] = APP_NAME
    swag['info']['description'] = APP_DESCRIPTION
    return jsonify(swag)


@app.before_request
def before_request():
    g.start_time = time.time()


@app.after_request
def after_request(response):
    diff = time.time() - g.start_time
    diff *= 1000.0  # to convert from seconds to milliseconds
    info = '%.4f ms <-- %s' % (diff, request.path)
    app.logger.info(info)
    return response


import error
import admin
import face_detect_api
