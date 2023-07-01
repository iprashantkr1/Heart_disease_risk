from flask import Flask,render_template,request
import joblib
import numpy as np

model=joblib.load('heart_risk_prediction_regression_model.sav')

app=Flask(__name__)

@app.route('/')
def index():

	return render_template('index.html')

@app.route('/about_us')
def about_us():

	return render_template('about_us.html')

@app.route('/contact_us')
def contact_us():

	return render_template('contact_us.html')

@app.route('/project_detail')
def project_detail():

	return render_template('project_detail.html')

@app.route('/patient_detail')
def patient_detail():

	return render_template('patient_detail.html')

@app.route('/result',methods=['POST'])
def result():

	result=request.form

	name=result['name']
	gender=float(result['gender'])
	age=float(result['age'])
	tc=float(result['tc'])
	hdl=float(result['hdl'])
	sbp=float(result['sbp'])
	smoke=float(result['smoke'])
	bpm=float(result['bpm'])
	diab=float(result['diab'])

	test_data=np.array([gender,age,tc,hdl,smoke,bpm,diab]).reshape(1,-1)
	prediction=abs(model.predict(test_data))

	resultDict={"name":name,"risk":round(prediction[0][0],2),"tc":tc,"hdl":hdl,"sbp":sbp}

	return render_template('result.html',results=resultDict)

if __name__ == "__main__":
	app.run(debug=True)
