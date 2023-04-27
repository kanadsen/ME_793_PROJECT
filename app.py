#import streamlit as st
import numpy as np
import streamlit as st
#from plotly.subplots import make_subplots
#import plotly.graph_objects as go


# set default page config to wide
st.set_page_config(layout="wide")

# Title text of the app
title_txt = "WEBAPP TO PREDICT BOILING POINT OF A MOLECULE"
st.markdown(f"<h3 style='text-align: center;'>{title_txt}</h1> <br><br>",unsafe_allow_html=True)

from main_ECC_file import*

model=torch.load('ECC_model.pt')
model.eval()
smiles_data=''
correct_boiling_point=0
c1=0
col1,col2,col3 = st.columns([2,1,2])
with col1:
   st.write('### INPUT PROPERTIES')
   smiles_data = str(st.text_input('Enter the Smiles of Molecule','O=C(Cl)OC(=C)C'))
   correct_boiling_point = float(st.text_input('Enter the correct Boiling Point (If known)',0))
   c1=correct_boiling_point
   correct_boiling_point=(correct_boiling_point-mean)/std
   correct_boiling_point=(correct_boiling_point-min_val)/(max_val-min_val)

mols=Chem.MolFromSmiles(smiles_data, sanitize=False)
with col3:   
   st.write("### Molecular structure")
   mols_list=[mols]
   #print(mols)
   img =  Draw.MolsToGridImage(mols_list, molsPerRow=1,legends=[f'Molecule : {smiles_data}'], returnPNG=False, subImgSize=(400, 400))#.save("img.png")#, legends=legends
   st.image(img)  
   df_1=pd.DataFrame()
   df_1['Smiles']=[smiles_data for i in range(100)]
   df_1['Tb']=[correct_boiling_point for i in range(100)]
   print(df_1)
   #smiles_data=np.array([smiles_data])
   test_mols = make_mol(df_1)
   print(test_mols)
   test_X = make_vec(test_mols)
   #print(model(test_X))
   test_loader = DataLoader(test_X, batch_size=len(test_X), shuffle=False)
   test_score,model,y_score,y_test=test(model, device,test_loader, args)
   y_score=np.array(y_score)
   y_score=y_score*(max_val-min_val)+min_val
   y_score=y_score*std+mean
   print(y_score)
   #y_score[0][0]=np.sum(np.array(y_score)) # Changed
   y_score[0][0] = round(y_score[0][0],3)
   #st.write(f'Boiling point {y_score[0]}')
   mae = abs(y_score[0][0] - c1 )
   del df_1 

col1,col2 = st.columns([2,2])
with col1:
   st.write(f"### OUTPUT GENERATED")
   st.text_area(label ="#### Predicted Boiling Point (K)",value='%.4f'%(y_score[0][0]), height=10)
   st.text_area(label = "#### Mean Absolute Error", value='%.4f'%(mae),height=10)



