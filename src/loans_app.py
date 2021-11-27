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
cat_features = ['Car_Owned','Bike_Owned','Active_Loan','House_Own',
                'Mobile_Tag','Homephone_Tag','Workphone_Working','Cleint_City_Rating']
num_features = list(set(x_columns) - set(cat_features))

def predict_price(ownedcredit, age, genderType, married, vehicleown):

  #  inputs_dict = {'KM_Driven' : float(kmDriven), 
          
   # df = pd.DataFrame(inputs_dict, index = [0])
    #print(df)

    #inputs = {c: df[c].values for c in df.columns}
    #for c in num_features:
      #  inputs[c] = inputs[c].astype(np.float32)
    #for k in inputs:
       # inputs[k] = inputs[k].reshape((inputs[k].shape[0], 1))            
  
    #sess = rt.InferenceSession(artifact_dir + '/usedcar_xgboost.onnx')
    #pred_onx = sess.run(None, inputs)

    predicted_score = float(500)
    return {f'Expected score {np.round(predicted_price, predicted_score)}' }


ownedcredit = gr.inputs.Slider(minimum=10, maximum=500, label="Credit Points")
age = gr.inputs.Dropdown(list(range(18, 100)), default="1", label="Age of the Credit Taker ((in years): ")
genderType = gr.inputs.Radio(['Male', 'Female', 'Others;'], default="Female", label="Client Gender")
married = gr.inputs.Radio(['Single', 'Married'], default="Single", label="Marriage Status")
vehicleown = gr.inputs.Radio(['4 Wheeler', '2 Wheeler', 'Both','None'], default="None", label="Vehicle Owner Type")

gr.Interface(predict_load, [ownedcredit, age, genderType, married, vehicleown], "text", live=False).launch(debug=True);