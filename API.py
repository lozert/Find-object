from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from model import predict, model_name

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()

class Imagelist(Resource):
    
    def get(self):
        return {"model_name" : model_name }, 200

    def post(self):
        url = request.args.get("url")

        if url:
            json_answer = predict(url)
            print(json_answer)
            return json_answer, 200
        else:
            return {"status": "400"}, 400


api.add_resource(Imagelist, '/model/')

if __name__ == "__main__":
    app.run(debug=True)

#curl -d "{\"url\":\"http://images.china.cn/attachement/jpg/site1005/20130115/001372a9a88e125f13a82d.jpg\"}" -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/process_image/