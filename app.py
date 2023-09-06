import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

df_1=pd.read_excel("C:\\Users\\Pushkar\\Desktop\\assignments\\SunBase Data\\customer_churn_large_dataset.xlsx")

q = ""

@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    
# Age, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB, Gender, Location
    

    
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    

    model = pickle.load(open("model.sav", "rb"))
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6]]
    
    new_df = pd.DataFrame(data, columns = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB', 'Gender', 'Location'])
                                           
    df_2 = pd.concat([df_1, new_df], ignore_index = True) 
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    #drop column customerID and tenure
    df_2.drop(columns= ['tenure'], axis=1, inplace=True)   
    
    
    
    
    new_df__dummies = pd.get_dummies(df_2[['Gender', 'Location']])

    single = model.predict(new_df__dummies.tail(1))
    probablity = model.predict_proba(new_df__dummies.tail(1))[:,1]
    
    if single==1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probablity*100)
        
    return render_template('home.html', output1=o1, output2=o2, 
                           query1 = request.form['query1'], 
                           query2 = request.form['query2'],
                           query3 = request.form['query3'],
                           query4 = request.form['query4'],
                           query5 = request.form['query5'], 
                           query6 = request.form['query6'])
    
app.run()