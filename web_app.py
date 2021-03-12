import tornado.web
import tornado.ioloop
from tornado.options import define, options
import os
from joblib import dump, load
from skimage.io import imread
from model.weather_net import WeatherNet
import cv2
import torch
import math
import uuid

define("port", default=8886)

def load_model(model_path='weathernet.pkl'):
    path = os.path.abspath(model_path)
    return load(path)

def soft_max(predict):
    probability = []
    sum = 0
    for i in predict:
        sum = sum + math.exp(i)
    for i in predict:
        probability.append(math.exp(i) / sum)
    return probability

def prediction(picture_path = 'work_with_data/dataset2/cloudy2.jpg'):
    weather_model = load_model()
    picture = cv2.resize(imread(picture_path), (64, 64))
    predict = weather_model(torch.tensor(picture[None, ...]).float().permute(0, 3, 1, 2))[0]
    probabilities = soft_max(predict)
    return probabilities

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("predict.html", prediction = "     ", predicted_class = " ")

class PredictHandler(tornado.web.RequestHandler):
    def post(self):
        picture_path = self.get_argument("picture_path", default=None, strip=False)
        print(picture_path)
        try:
            predict = prediction(picture_path)
        except Exception as err:
            self.render("predict.html", prediction = "      ", predicted_class = " ")
        pred_index = predict.index(max(predict))
        dict_prediction = {1: 'cloudy', 2: 'rain', 3: 'shine', 4:'sunrise'}
        pred_class = dict_prediction.get(pred_index+1)
        self.render("predict.html", prediction = predict, predicted_class = pred_class)
    get = options = post

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler),
            (r"/predict", PredictHandler)
        ]
        settings = dict(
            title=u"Weather Predictor",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=False,
            cookie_secret= uuid.uuid4().int,
            debug=True,
        )
        super(Application, self).__init__(handlers, **settings)


if __name__ == "__main__":
    app = Application()
    app.listen(options.port)
    print('the game had started')
    tornado.ioloop.IOLoop.current().start()