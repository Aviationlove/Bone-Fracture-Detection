from flask import Flask
app=Flask(__name__)
@app.route("/")
def title():
    return "Medimage AI"
