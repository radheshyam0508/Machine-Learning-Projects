from flask import Flask, render_template, request
 
import pickle

app = Flask(__name__)


# open a file, where you ant to store the data
file = open('model.pkl', 'rb')
clf=pickle.load(file)
file.close()

@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method=="POST":
        #print(request.form)
        myDict=request.form
        bodyTemp=int(myDict['bodyTemp'])
        bodyPain=int(myDict['bodyPain'])
        runnyNose=int(myDict['runnyNose'])
        diffBreath=int(myDict['diffBreath'])
        o2Saturation=int(myDict['o2Saturation'])
        travelHistory=int(myDict['travelHistory'])
        age=int(myDict['age'])
        LossofTasteSmell=int(myDict['LossofTasteSmell'])
        vomiting=int(myDict['vomiting'])
        Diarrhea=int(myDict['Diarrhea'])
        # Code for inference
        inputFeatures= ([[bodyTemp,bodyPain,runnyNose,diffBreath,o2Saturation,travelHistory,age,LossofTasteSmell,vomiting,Diarrhea]])
        infProb=clf.predict_proba(inputFeatures)[0][1]
        #print(infProb)
        return render_template('show.html',inf=round(infProb*100))


    #return 'Hello, world Hari Om!'+ str(infProb)
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)