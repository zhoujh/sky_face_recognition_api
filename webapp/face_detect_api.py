VERSION_STR = 'v1.0.0'

import cv2
import uuid
import base64
import requests
import numpy as np
from error import Error
from flask import Blueprint, request, jsonify
from face_mark import detect_face
import json


blueprint = Blueprint(VERSION_STR, __name__)

def base64_encode_image(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ret, image_buf = cv2.imencode('.jpg', image_bgr, (cv2.IMWRITE_JPEG_QUALITY, 40))
    image_str = base64.b64encode(image_buf)
    return 'data:image/jpeg;base64,' + image_str

def obtain_images(request):
    '''
    All three routes below pass the image in the same way as one another.
    This function attempts to obtain the image, or it throws an error
    if the image cannot be obtained.
    '''

    if 'image_url' in request.args:
        image_url = request.args['image_url']
        try:
            response = requests.get(image_url)
            encoded_image_str = response.content
        except:
            raise Error(2873, 'Invalid `image_url` parameter')

    elif len(request.files) > 0:
        for filename in request.files:
            name = request.files[filename].name
            print(filename+'-----'+name)
            # image_buf = request.files[filename]  # <-- FileStorage object
            encoded_image_str = request.files[filename].read()
            image_name = filename
            if (image_name.find('.') > 0):
                id = '.'.join(image_name.split('.')[:-1])
            else:
                id = image_name

    elif 'image_base64' in request.args:
        image_base64 = request.args['image_base64']

        ext, image_str = image_base64.split(';base64,')
        try:
            encoded_image_str = base64.b64decode(image_str)
        except:
            raise Error(2873, 'Invalid `image_base64` parameter')

    else:
        raise Error(35842, 'You must supply either `image_url` or `image_buf`')

    if encoded_image_str == '':
        raise Error(5724, 'You must supply a non-empty input image')

    encoded_image_buf = np.fromstring(encoded_image_str, dtype=np.uint8)
    # decoded_image_bgr = cv2.imdecode(encoded_image_buf, cv2.IMREAD_COLOR)
    if 'id' in request.args:
        id = request.args['id']
    return id,encoded_image_buf

@blueprint.route('/detect_faces', methods=['POST'])
def detect_faces():
    '''
    Find faces and predict emotions in a photo
    Find faces and their emotions, and provide an annotated image and thumbnails of predicted faces.
    ---
    tags:
      - v1.0.0

    responses:
      200:
        description: A photo info objects
        schema:
          $ref: '#/definitions/ResultInfo'
      default:
        description: Unexpected error
        schema:
          $ref: '#/definitions/Error'

    parameters:
      - name: image_buf
        in: formData
        description: An image that should be processed. This is used when you need to upload an image for processing rather than specifying the URL of an existing image. If this field is not specified, you must pass an image URL via the `image_buf` parameter
        required: false
        type: file


    consumes:
      - multipart/form-data
      - application/x-www-form-urlencoded

    definitions:
      - schema:
            id: ResultInfo
            type: object
            properties:
                id:
                    type: string
                    format: byte
                    description: an identification number for received image
                result_image:
                    type: string
                    format: byte
                    description: a base64 encoded detected result image
    '''
    id,encoded_image_buf = obtain_images(request)
    photoinfo = detect_face(encoded_image_buf)
    # photoinfo['_id'] = str(photoinfo['_id']) # makes ObjectId jsonify
    response = jsonify(photoinfo)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@blueprint.route('/upload_faces', methods=['POST'])
def upload_faces():
    '''
    upload faces for face recognition
    ---
    tags:
      - v1.0.0

    responses:
      200:
        description: A photo info objects
        schema:
          $ref: '#/definitions/ResultInfo'
      default:
        description: Unexpected error
        schema:
          $ref: '#/definitions/Error'

    parameters:
      - name: id
        in: query
        description: id for face
        required: true
        type: string
      - name: image_form
        in: formData
        description: An image that should be processed. This is used when you need to upload an image in formData
        required: false
        type: file

    consumes:
      - multipart/form-data
      - application/x-www-form-urlencoded

    definitions:
      - schema:
          id: ResultInfo
          type: object
          required:
              id:
                type: string
                format: byte
                description: an identification number for received image
              result_image:
                type: string
                format: byte
                description: a base64 encoded detected result image
    '''
    id,encoded_image_buf = obtain_images(request)
    photoinfo = detect_face(encoded_image_buf)
    id  = save_face_image(id,encoded_image_buf)
    photoinfo['id'] = id
    response = jsonify(photoinfo)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

def save_face_image(id,encoded_image_buf):
    decoded_image_bgr = cv2.imdecode(encoded_image_buf, cv2.IMREAD_COLOR)
    # encoded_image_jpg = cv2.imencode('.jpg',encoded_image_buf)
    file_name = '../data/'+id+'.jpg'
    # cv2.imwrite('../data/'+id+'.jpg',decoded_image_bgr,dtype=np.uint8)
    cv2.imencode('.jpg', decoded_image_bgr)[1].tofile(file_name)
    print('save to '+file_name);
    return id;
from app import app
app.register_blueprint(blueprint, url_prefix='/'+VERSION_STR)
