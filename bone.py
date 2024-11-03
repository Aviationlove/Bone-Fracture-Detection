from flask import Flask
app=Flask(__name__)
@app.route("/")
def title():
    return "Medimage AI"
if __name__=='__main__':
    app.run(debug=False, host='0.0.0.0')