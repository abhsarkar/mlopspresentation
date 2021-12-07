import os
import wandb
import pandas as pd
import numpy as np
import onnxruntime as rt
from gradio import gradio as gr
#os.environ["WANDB_API_KEY"] = "526145026d9030cfd5d66bc0f200786831a5b80a"
#run = wandb.init(project='usedcar')
#ARTIFACT_NAME = 'auto_loan:v0'
#artifact = run.use_artifact('abhsarkar/auto_loan/' + ARTIFACT_NAME, type='model')
#artifact_dir = artifact.download()

x_columns = ['Client_Gender','Client_Occupation','Car_Owned','Client_Education','Active_Loan','Cleint_City_Rating','Workphone_Working',
             'House_Own','Type_Organization','Client_Income_Type','Registration_Days','Child_Count','Credit_Amount']


cat_features = ['Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own',
                'Mobile_Tag', 'Homephone_Tag', 'Workphone_Working', 'Cleint_City_Rating']

num_features = list(set(x_columns) - set(cat_features))
def predict_price(clientGender, clientOccupation, carOwn,
                  clientEdu, activeLoan, clientCity, 
                  workPhone, ownHouse, orgType, 
                  incomeType, regDays, children,amntNeed):

  
    inputs_dict = {'Client_Gender' : float(clientGender), 
              'Client_Occupation': float(clientOccupation), 
              'Car_Owned': float(carOwn), 
              'Client_Education': float(clientEdu), 
              'Active_Loan': float(activeLoan), 
              'Cleint_City_Rating': float(clientCity), 
              'Workphone_Working': float(workPhone), 
              'House_Own': float(ownHouse), 
              'Type_Organization': float(orgType), 
              'Client_Income_Type': float(incomeType), 
              'Registration_Days': float(regDays), 
              'Child_Count': float(children),
              'Credit_Amount': float(amntNeed)}
    

    df = pd.DataFrame(inputs_dict, index = [0])
    print(df)

    inputs = {c: df[c].values for c in df.columns}
    for c in num_features:
        inputs[c] = inputs[c].astype(np.float32)
    for k in inputs:
        inputs[k] = inputs[k].reshape((inputs[k].shape[0], 1))            
  
    sess = rt.InferenceSession(artifact_dir + '../artifacts/autoloan_xgboost.onnx')
    pred_onx = sess.run(None, inputs)

    predicted_price = float(pred_onx[0][0,0]) }


activeLoan = gr.inputs.Slider(minimum=10, maximum=500, label="activeLoan  Lakhs")
clientGender = gr.inputs.Radio(['Male', 'Female', 'Others;'], default="Female", label="Client Gender")
amntNeed = gr.inputs.Radio(['50000', '5000000', '1000000','7500000'], default="0", label="Amount Needed")

gr.Interface(predict_score, [clientGender,
                          activeLoan,
                          amntNeed,
                          clientEdu=float(2), 
                          clientOccupation=float(4),
                        carOwn=float(8)
                          clientCity=4, 
                          workPhone=5, 
                          ownHouse=1,
                          orgType=55, 
                          incomeType=1000.0, 
                          regDays=185.0, 
                          children=4
                          ], "text",live=False).launch(debug=True,share=true);