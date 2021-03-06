from flask import Flask
from flask_cors import CORS
from flask_restful import Api


from config import Config
from model import DB
from resources.knn import *
from resources.naive_bayes import *
from resources.id3 import *
from resources.training_overall import *
from resources.predict_overall import *
from resources.training_sample import *
from resources.predictSample import *


app = Flask(__name__)
app.config.from_object(Config)
CORS(app)
api = Api(app)

DB.init(
        '127.0.0.1:27017',
        '',
        '',
        '',
        "khoa_luan",
    )
# @app.before_first_request


api.add_resource(TrainingOverall, "/training-overall")
api.add_resource(PredictOverall, "/predict-overall")
api.add_resource(Load_Scrore_Table, "/load_score")
# api.add_resource(Knn, "/knn")
api.add_resource(NaiveBayes, "/naive")
api.add_resource(ID3, "/id3")
api.add_resource(TrainNaive, "/train-naive")
api.add_resource(TrainNaiveTest, "/train-naive-test")
api.add_resource(TrainID3Test, "/train-id3-test")
api.add_resource(RecommendSystem, "/recommend")
api.add_resource(Preprocessing_Predict_Data, "/preprocess")
api.add_resource(TrainingSample, "/train-sample")
api.add_resource(PredictSample, "/predict-sample")

# api.add_resource(User, "/user", "/user/<string:ma>")

# api.add_resource(
#     Student_And_Class, "/danhsach",
#                        "/danhsach/<int:id_lop>",
#                        "/danhsach/<int:id_lop>",
#                        "/danhsach/<int:id_lop>/<string:ma>"
# )

# api.add_resource(Teacher_And_Class, "/teach", "/teach/<int:id_lop>/<string:ma>", "/teach/<int:id_lop>")
# api.add_resource(UploadAva, "/upload/<string:ma>", "/upload/<string:ma>/<string:filename>")

if __name__ == "__main__":
    app.run(port=5000, debug=True)