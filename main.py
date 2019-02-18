from flask_api import FlaskAPI

app = FlaskAPI(__name__)


@app.route('/')
def hello():
    return {"Hello": "World"}

#
# if __name__ == "__main__":
#     app.run()


