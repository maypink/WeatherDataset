import tornado.web
import tornado.ioloop
from tornado.options import define, options
import os
from model.weather_net import WeatherNet
import torch
import uuid
import numpy as np
from scipy.special import softmax
from PIL import Image
import io
import json

define("port", default=8883)

def load_model(model_path):
    path = os.path.abspath(model_path)
    model = WeatherNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("predict.html", prediction = "     ", predicted_class = " ")


class PredictHandler(tornado.web.RequestHandler):
    def initialize(self, weather_model):
        self.weather_model = weather_model

    def picture_prep(self, pic):
        pic = pic.resize((64, 64))
        pic = np.array(pic)
        picture_tensor = torch.tensor(pic[None, ...]).float().permute(0, 3, 1, 2)
        return picture_tensor

    def post(self):
        picture = self.request.files['picture'][0]
        img = Image.open(io.BytesIO(picture['body']))
        img = self.picture_prep(img)
        try:
            predict = self.weather_model.prediction(img)
        except Exception as err:
            self.render("predict.html", prediction="      ", predicted_class=" ")
        pred_index = predict.index(max(predict))
        dict_prediction = {1: 'cloudy', 2: 'rain', 3: 'shine', 4:'sunrise'}
        pred_class = dict_prediction[pred_index+1]
        self.write(json.dumps({'Distribution of probabilities': str(predict[0]), 'Predicted class': pred_class }))
        #self.render("predict.html", prediction = predict, predicted_class = pred_class)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler),
            (r"/predict", PredictHandler, dict(weather_model=weather_model)),
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
    model_path = '../weights/weathernet.pt'
    weather_model = load_model(model_path)
    app = Application()
    app.listen(options.port)
    print('the game had started')
    tornado.ioloop.IOLoop.current().start()
