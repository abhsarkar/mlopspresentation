ownedcredit = gr.inputs.Slider(minimum=10, maximum=500, label="Credit Points")
age = gr.inputs.Dropdown(list(range(18, 100)), default="1", label="Age of the Credit Taker ((in years): ")
genderType = gr.inputs.Radio(['Male', 'Female', 'Others;'], default="Female", label="Client Gender")
married = gr.inputs.Radio(['Single', 'Married'], default="Single", label="Marriage Status")
vehicleown = gr.inputs.Radio(['4 Wheeler', '2 Wheeler', 'Both','None'], default="None", label="Vehicle Owner Type")

gr.Interface(predict_load, [ownedcredit, age, genderType, married, vehicleown], "text", live=False).launch(debug=True);