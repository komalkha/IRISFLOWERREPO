from flask import Flask,render_template,request
import config
from utils import Iris

app = Flask(__name__)

@app.route('/')
def Welcome():
    print('Welcome')
    return render_template('home.html')

@app.route('/predict')
def predict():
    SepalLengthCm = 5.6
    SepalWidthCm = 2.2
    PetalLengthCm = 5.5
    PetalWidthCm = 1.7
    result = Iris(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm).predict_flower()
    return f'{result}'

@app.route('/user')
def user_input ():
    data = request.form  
    SepalLengthCm = eval(data['SepalLengthCm'])
    SepalWidthCm = eval(data['SepalWidthCm'])
    PetalLengthCm = eval(data['PetalLengthCm'])
    PetalWidthCm = eval(data['PetalWidthCm'])
    iris1 = Iris(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
    result = iris1.predict_flower()
    return f'{result}'
if __name__ == '__main__' :
    app.run(port = config.PORT_NO,debug=True)