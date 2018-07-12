from flask import Flask
from flask import request
from flask import jsonify
from predict import predict as pred

application = Flask(__name__)
 
@application.route('/', methods=['GET', 'POST'])
def resolve():
    #print("received something")
    print("Question is: "+request.form['question'])
    f = request.files['image']
    f.save(f.filename)
    
    return jsonify(predictStuff(request.form['question'],f.filename))
 
def predictStuff(qns,file):
    return pred(file,qns)
#    pred = [('Apple',0.43),('Pear',0.12)]
    
#    return {'pred': pred}

if __name__ == '__main__':
    application.run(host='0.0.0.0')
