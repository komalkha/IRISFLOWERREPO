import numpy as np
import pandas as pd
import config
import json
import pickle


class Iris():
    def __init__(self,SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):
        self.SepalLengthCm = SepalLengthCm
        self.SepalWidthCm = SepalWidthCm
        self.PetalLengthCm = PetalLengthCm
        self.PetalWidthCm = PetalWidthCm

    def load_model(self):
        with open (config.MODEL_FILE_PATH,'rb') as f:
            self.model = pickle.load(f)
    
        with open (config.JSON_FILE_PATH,'r') as f:
            self.columns = json.load(f)
    
    def predict_flower(self):
        self.load_model()
        array = np.zeros(4)     # len(self.columns)
        array[0] = self.SepalLengthCm 
        array[1] = self.SepalWidthCm 
        array[2] = self.PetalLengthCm
        array[3] = self.PetalWidthCm 
        print(array)
        flower = self.model.predict([array])[0]
        print(f'Flower for given feature is {flower}')
        return f'Flower Predicted will be : {flower.upper()}'


if __name__ == "__main__":
    SepalLengthCm = 5.6
    SepalWidthCm = 2.2
    PetalLengthCm = 5.5
    PetalWidthCm = 1.7
    iris1 = Iris(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
    iris1.predict_flower
