import tornado.web
import tornado.ioloop
from tornado.options import define, options
import os
from skimage.io import imread
from model.weather_net import WeatherNet
import cv2
import torch
import math
import uuid
from scipy.special import softmax

define("port", default=8886)

def load_model(model_path='../weathernet.pt'):
    path = os.path.abspath(model_path)
    model = WeatherNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def prediction(picture):
    global weather_model
    predict = weather_model(picture)[0]
    probabilities = list(softmax(predict.detach().numpy()))
    return probabilities

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("predict.html", prediction = "     ", predicted_class = " ")

class PredictHandler(tornado.web.RequestHandler):
    def picture_prep(self, picture_path):
        picture = cv2.resize(imread(picture_path), (64, 64))
        picture_tensor = torch.tensor(picture[None, ...]).float().permute(0, 3, 1, 2)
        return picture_tensor

    def post(self):
        picture_path = self.get_argument("picture_path", default=None, strip=False)
        picture = self.picture_prep(picture_path)
        try:
            predict = prediction(picture)
        except Exception as err:
            self.render("predict.html", prediction = "      ", predicted_class = " ")
        pred_index = predict.index(max(predict))
        dict_prediction = {1: 'cloudy', 2: 'rain', 3: 'shine', 4:'sunrise'}
        pred_class = dict_prediction[pred_index+1]
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
    weather_model = load_model()
    print('the game had started')
    tornado.ioloop.IOLoop.current().start()
