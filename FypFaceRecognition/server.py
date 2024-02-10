from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from RecognizeFace import FaceRecognizer
import os
from CaputerData import CaptureData
import requests


app = Flask(__name__)
CORS(app, origins="*", allow_headers="*", methods="*")
app.debug = True

@app.route('/admin-student/picture', methods=['POST'])
@cross_origin()
def upload_file():
    data = request.get_json()
    id=data['id']
    if os.path.isfile(f'images/{id}.png'):
        print('Id Exists')
        return jsonify({'message': 'Id already exists','statusCode': 400})
    
    captureData=CaptureData(id)
    captureData.capture()
    return jsonify({'message': 'Image saved', 'statusCode': 200})

@app.route('/recognize-and-add-attendance', methods=['POST'])
def recognize_and_add_attendance():
    recognizer = FaceRecognizer()
    result=recognizer.recognize()
    print('This is Result: ', result)
    for i in range(len(result)):
        print(result[i])
        time=result[i]['entryTime']
        hour, minute = time.split(':')[:2]
        if int(hour) > 12:
            hour = int(hour) - 12
        print('This is Hour[0]', hour)
        time = f"{hour}:{minute}" 
        data = {'time': time}  
        college_id = result[i]['collegeId']
        response = requests.post(f'http://localhost:3000/teacher/add-student-attendance/{college_id}', json=data)
        print('This is Response: ', response.json())
        
    # for college_id in result:
    #     response=requests.post(f'http://localhost:3000/teacher/add-student-attendance/{college_id}')
    #     print('This is Response: ', response.json())
if __name__ == '__main__':
    app.run(port=3002)