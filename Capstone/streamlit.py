import streamlit as st
import pandas as pd

import pickle


model_file = pickle.load(open(st.secrets["file_pkl"],'rb'))


cap_shape_dict = {"b":0,"c":1,"f":2,"k":3,"s":4,"x":5}
cap_surface_dict = {"f":0,"g":1,"s":2,"y":3}
cap_color_dict = {"b":0,"c":1,"e":2,"g":3,"n":4,"p":5,"r":6,"u":7,"w":8,"y":9}
bruises_dict = {"f":0,"t":1}
odor_dict = {"a":0,"c":1,"f":2,"l":3,"n":5,"p":6,"s":7,"y":8}
gill_attachment_dict = {"f":1}
gill_spacing_dict = {"c":0,"w":1}
gill_size_dict = {"b":0,"n":1}
gill_color_dict = {"b":0,"e":1,"g":2,"h":3,"k":4,"n":5,"p":7,"r":8,"u":9,"w":10}
stalk_shape_dict = {"e":0,"t":1}
stalk_root_dict = {"?":0,"b":1,"c":2,"e":3,"r":4}
stalk_surface_above_ring_dict = {"f":0,"k":1, "s":2}
stalk_surface_below_ring_dict = {"f":0,"k":1, "s":2, "y":3}
stalk_color_above_ring_dict = {"b":0,"e":2,"g":3,"n":4,"p":6,"w":7}
stalk_color_below_ring_dict = {"b":0,"e":2,"g":3,"n":4,"p":6,"w":7, "y":8}
veil_type_dict = {"p":0}
veil_color_dict = {"w":2}
ring_number_dict = {"o":1,"t":2}
ring_type_dict = {"e":0,"f":1,"l":2,"p":4}
spore_print_color_dict = {"h":1,"k":2,"n":3,"r":5,"u":6,"w":7}
population_dict = {"a":0,"c":1,"n":2,"s":3,"v":4,"y":5}
habitat_dict = {"d":0,"g":1,"l":2,"m":3,"p":4,"u":5, "w":6}

columns_X = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']

# with open(model_file, 'rb') as f_in:
#     model = pickle.load(f_in)


def predict(answers_dict):
    df = pd.DataFrame(answers_dict.items()).T
    df.columns = columns_X
    
    df['cap-shape'] = df['cap-shape'].map(cap_shape_dict)
    df['cap-surface'] = df['cap-surface'].map(cap_surface_dict)
    df['cap-color'] = df['cap-color'].map(cap_color_dict)
    df['bruises'] = df['bruises'].map(bruises_dict)
    df['odor'] = df['odor'].map(odor_dict)
    df['gill-attachment'] = df['gill-attachment'].map(gill_attachment_dict)
    df['gill-spacing'] = df['gill-spacing'].map(gill_spacing_dict)
    df['gill-size'] = df['gill-size'].map(gill_size_dict)
    df['gill-color'] = df['gill-color'].map(gill_color_dict)
    df['stalk-shape'] = df['stalk-shape'].map(stalk_shape_dict)
    df['stalk-root'] = df['stalk-root'].map(stalk_root_dict)
    df['stalk-surface-above-ring'] = df['stalk-surface-above-ring'].map(stalk_surface_above_ring_dict)
    df['stalk-surface-below-ring'] = df['stalk-surface-below-ring'].map(stalk_surface_below_ring_dict)
    df['stalk-color-above-ring'] = df['stalk-color-above-ring'].map(stalk_color_above_ring_dict)
    df['stalk-color-below-ring'] = df['stalk-color-below-ring'].map(stalk_color_below_ring_dict)
    df['veil-type'] = df['veil-type'].map(veil_type_dict)
    df['veil-color'] = df['veil-color'].map(veil_color_dict)
    df['ring-number'] = df['ring-number'].map(ring_number_dict)
    df['ring-type'] = df['ring-type'].map(ring_type_dict)
    df['spore-print-color'] = df['spore-print-color'].map(spore_print_color_dict)
    df['population'] = df['population'].map(population_dict)
    df['habitat'] = df['habitat'].map(habitat_dict)

    prediction = model.predict(df.iloc[-1:])
    
    return prediction.astype(int)


st.write("""
# My First App
Hello *world!*
"""
)

answers_dict = {}

expander = st.expander("Mushroom information")

with expander:
    answers_dict['cap-shape'] = st.selectbox('cap-shape?', ['b', 'c', 'f', 'k', 's', 'x'])
    answers_dict['cap-surface'] = st.selectbox('cap-surface?', ['f', 'g', 's', 'y'])
    answers_dict['cap-color'] = st.selectbox('cap-color?', ['b', 'c', 'e', 'g', 'n', 'p', 'r', 'u', 'w', 'y'])
    answers_dict['bruises'] = st.selectbox('bruises?', ['f', 't'])
    answers_dict['odor'] = st.selectbox('odor?', ['a', 'c', 'f', 'l', 'n', 'p', 's', 'y'])
    answers_dict['gill-attachment'] = st.selectbox('gill-attachment?', ['f'])
    answers_dict['gill-spacing'] = st.selectbox('gill-spacing?', ['c', 'w'])
    answers_dict['gill-size'] = st.selectbox('gill-size?', ['b', 'n'])
    answers_dict['gill-color'] = st.selectbox('gill-color?', ['b', 'e', 'g', 'h', 'k', 'n', 'p', 'r', 'u', 'w'])
    answers_dict['stalk-shape'] = st.selectbox('cap-shape?', ['e', 't'])

    answers_dict['stalk-root'] = st.selectbox('stalk-root?', ['?', 'b', 'c', 'e', 'r'])
    answers_dict['stalk-surface-above-ring'] = st.selectbox('stalk-surface-above-ring?', ['f', 'k', 's'])
    answers_dict['stalk-surface-below-ring'] = st.selectbox('stalk-surface-below-ring?', ['f', 'k', 's', 'y'])
    answers_dict['stalk-color-above-ring'] = st.selectbox('stalk-color-above-ring?', ['b', 'e', 'g', 'n', 'p', 'w'])
    answers_dict['stalk-color-below-ring'] = st.selectbox('stalk-color-below-ring?', ['b', 'e', 'g', 'n', 'p', 'w', 'y'])
    answers_dict['veil-type'] = st.selectbox('veil-type?', ['p'])
    answers_dict['veil-color'] = st.selectbox('veil-color?', ['w'])
    answers_dict['ring-number'] = st.selectbox('ring-number?', ['o', 't'])
    answers_dict['ring-type'] = st.selectbox('ring-type?', ['e', 'f', 'l', 'p'])
    answers_dict['spore-print-color'] = st.selectbox('spore-print-color?', ['h', 'k', 'n', 'r', 'u', 'w'])

    answers_dict['population'] = st.selectbox('population?', ['a', 'c', 'n', 's', 'v', 'y'])
    answers_dict['habitat'] = st.selectbox('habitat?', ['d', 'g', 'l', 'm', 'p', 'u', 'w'])

    

    

if st.button('Predict Value'):
    value = predict(answers_dict)
    value_str = 'Poison' if value == 1 else 'Edible' 
    st.write(f'The mushroom is {value_str} mushroom.')
    #st.write(predict(answers_dict))
    #st.write(answers_dict)
    #st.write(pd.DataFrame(answers_dict.items()).T)