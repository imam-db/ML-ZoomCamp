import pandas as pd
import pickle
from flask import Flask, render_template, request
import os

app = Flask(__name__)
filename = 'mushroom_model.pkl'
model = pickle.load(open(filename, 'rb'))
port = int(os.environ.get("PORT", 5000))

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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('upload.html')
    elif request.method == 'POST':
        csv_file = request.files.get('file')
        X_test = pd.read_csv(csv_file)
        X_test_view = X_test.copy()
        X_test['cap-shape'] = X_test['cap-shape'].map(cap_shape_dict)
        X_test['cap-surface'] = X_test['cap-surface'].map(cap_surface_dict)
        X_test['cap-color'] = X_test['cap-color'].map(cap_color_dict)
        X_test['bruises'] = X_test['bruises'].map(bruises_dict)
        X_test['odor'] = X_test['odor'].map(odor_dict)
        X_test['gill-attachment'] = X_test['gill-attachment'].map(gill_attachment_dict)
        X_test['gill-spacing'] = X_test['gill-spacing'].map(gill_spacing_dict)
        X_test['gill-size'] = X_test['gill-size'].map(gill_size_dict)
        X_test['gill-color'] = X_test['gill-color'].map(gill_color_dict)
        X_test['stalk-shape'] = X_test['stalk-shape'].map(stalk_shape_dict)
        X_test['stalk-root'] = X_test['stalk-root'].map(stalk_root_dict)
        X_test['stalk-surface-above-ring'] = X_test['stalk-surface-above-ring'].map(stalk_surface_above_ring_dict)
        X_test['stalk-surface-below-ring'] = X_test['stalk-surface-below-ring'].map(stalk_surface_below_ring_dict)
        X_test['stalk-color-above-ring'] = X_test['stalk-color-above-ring'].map(stalk_color_above_ring_dict)
        X_test['stalk-color-below-ring'] = X_test['stalk-color-below-ring'].map(stalk_color_below_ring_dict)
        X_test['veil-type'] = X_test['veil-type'].map(veil_type_dict)
        X_test['veil-color'] = X_test['veil-color'].map(veil_color_dict)
        X_test['ring-number'] = X_test['ring-number'].map(ring_number_dict)
        X_test['ring-type'] = X_test['ring-type'].map(ring_type_dict)
        X_test['spore-print-color'] = X_test['spore-print-color'].map(spore_print_color_dict)
        X_test['population'] = X_test['population'].map(population_dict)
        X_test['habitat'] = X_test['habitat'].map(habitat_dict)
        X_test['class'] = model.predict(X_test)
        X_test['class'] = X_test['class'].map({0:'edible', 1:'poison'})
        X_test_view['class'] = X_test['class']
        return X_test_view.to_html()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=port)
    #serve(app, host='0.0.0.0', port=port)