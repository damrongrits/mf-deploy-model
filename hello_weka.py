

from flask import Flask, request
import weka.core.serialization as serialization
from weka.classifiers import Classifier
from weka.core.converters import Loader, Saver
from weka.core.dataset import Attribute, Instance, Instances
import os;
import weka.core.jvm as jvm
#export JAVA_HOME="/Library/Java/JavaVirtualMachines/jdk-11.0.10.jdk/Contents/Home"
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-11.0.10.jdk/Contents/Home"
jvm.start()

app = Flask(__name__)

@app.route("/hello_world")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/prediction', methods = ['POST', 'GET'])
def prediction():
    if request.method == 'GET':
        return 'Please Resend with POST Method with Attributes'

    elif request.method == 'POST':
        X = request.get_json()
        
        x1 = X['Type']
        x2 = X['Air temperature [K]']
        x3 = X['Process temperature [K]']
        x4 = X['Rotational speed [rpm]']
        x5 = X['Torque [Nm]']
        x6 = X['Tool wear [min]']


        
        objects = serialization.read_all("PMj48.model")
        classifier = Classifier(jobject=objects[0])
        loader = Loader (classname = "weka.core.converters.ArffLoader")
        # create attributes
        type_att = Attribute.create_nominal("Type", ["M", "L", "H"])
        num1_att = Attribute.create_numeric("Air temperature [K]")
        num2_att = Attribute.create_numeric("Process temperature [K]")
        num3_att = Attribute.create_numeric("Rotational speed [rpm]")
        num4_att = Attribute.create_numeric("Torque [Nm]")
        num5_att = Attribute.create_numeric("Tool wear [min]")
        MF_att = Attribute.create_nominal("MF", ["Normal", "Failure"])
        # create dataset
        dataset = Instances.create_instances("helloworld", [type_att, num1_att, num2_att, num3_att, num4_att, num5_att, MF_att], 0)
        dataset.class_is_last()
        # add rows
        #values = [1,300.7,311.9,1335,57.1,194,1]
        values = [x1,x2,x3,x4,x5,x6,0]
        inst = Instance.create_instance(values)
        dataset.add_instance(inst)

        predicted = classifier.classify_instance(dataset[0])
        #jvm.stop()
        return  str(predicted)

if __name__ == '__main__':
    app.run(debug = True)