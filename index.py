from flask import Flask, render_template, request
from forms import Fotos
#from flask.wrappers import Request
####ML###
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

from werkzeug.utils import secure_filename
import flask
import io
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils

app = Flask(__name__)
app.secret_key="666666666"
model = None
#rutas
@app.route('/') #pagina principal
def home():
    form = Fotos()
    return render_template('home.html', form=form)

#about
@app.route('/about') #pagina principal
def about():
    #return 'Sobre la página'
    return render_template('about.html')

#formulario
@app.route('/formulario', methods=['POST','GET']) #pagina principal
def form_get():
    form = Fotos()
    if request.method == "GET":
        return render_template('home.html')
    elif request.method == "POST":
        data = request.form.to_dict()
        print(data)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        #dir = os.path.abspath(os.path.dirname(__file__))
        #main_dir = "/data"
        train_dir = "data/test"
        #upload_dir = "../../src/data/test"
        upload_dir = "../../src/static/uploads"
        #path = os.path.join(main_dir,train_dir)
        path = os.path.join(dir_path,train_dir)
        #os.listdir(path)

        #cambiar la foto


        ##
    
        #file = request.files['file']
        #filename = secure_filename(file.filename)
        #file.save(os.path.join('uploads', filename))
        ##return redirect(url_for('prediction', filename=filename))

        ###probar online
        print("::5::")
        ##image = flask.request.files['foto'].read()
        ##image = Image.open(io.BytesIO(image))





        ##print("::6::")
        ##image.convert("L")
        ##image = image.resize((80, 80))
        ##image = img_to_array(image)
        ##image = np.expand_dims(image, axis=0)
        ##image = imagenet_utils.preprocess_input(image)

        ##image = np.array(image).reshape(-1,80,80,1)
        ##image = image/255

        ##
        ##print("::1::")
        uploaded_file = request.files['foto']
        #filename = secure_filename(uploaded_file.filename)
        if uploaded_file != '':
            #uploaded_file.save(uploaded_file.filename)
            print("::2::")
            uploaded_file.save(os.path.join(upload_dir, uploaded_file.filename))
            print("::3::")
        #img = cv2.imread(uploaded_file.filename,0)
        ##img_array = cv2.imread(uploaded_file.filename,cv2.IMREAD_GRAYSCALE)
        img_array = cv2.imread(os.path.join(upload_dir, uploaded_file.filename),cv2.IMREAD_GRAYSCALE)
        
        #img = request.files['foto']
        print("::7::")
        


        #data =request.files['file']
        img = Image.open(request.files['foto'])
        img = np.array(img)
        #img = cv2.resize(img,(224,224))
        #img_array = cv2.cvtColor(np.array(img), cv2.IMREAD_GRAYSCALE)




        p="1.jpg"
        ##p=filename
        ##print(":::",filename,":::")

        category = p.split(".")[0]
        #img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        ##img_array = cv2.imread(np.array(img),cv2.IMREAD_GRAYSCALE)
        
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        #plt.imshow(new_img_array,cmap="gray")
            
            
        #create_test1_data(path)
        new_img_array = np.array(new_img_array).reshape(-1,80,80,1)
        new_img_array = new_img_array/255
        ##print(new_img_array)

        predictions = model.predict(new_img_array)





        #predictions = model.predict(image)
        print("::8::",predictions,"::::")
        #predicted_val = [int(round(p[0])) for p in predictions]
        predicted_val = [int(round(p[0])) for p in predictions]
        print("::9::")
        ##submission_df = pd.DataFrame({'id':category, 'label':predicted_val})
        print("::",predicted_val,"::")
        if (predicted_val == [1] ):
            print ("Creo que es un gato")
            salida="Creo que es un gato"
        else: 
            print ("Creo que es un perro")
            salida="Creo que es un perro"
        print("\n___________\n")
        ##print(submission_df)


        return render_template('answer.html', resultado=salida, foto=uploaded_file.filename)

        #return salida
        #cv2.imshow("Outlined #2", img_outlined2)
        #cv2.waitKey(0)                                                                          # Wait until a key is pressed.
        #cv2.destroyAllWindows()   

        #return f"username={ request.form['nombre']},foto={ request.form['foto']}"
        #return render_template("home.html", form=form)
    else:
        return "metodo no aceptado"


#escuchar siempre
if __name__ == '__main__':
    
    



    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.abspath(os.path.dirname(__file__))
    #cwd = os.getcwd()
    #print("---------------------")
    #print(dir_path)
    #print(cwd)
    #print(dir)
    #print("---------------------")
    dir_model='data/saved_model/my_model'
    #print("---------------------")
    #print(dir_path)
    #print(os.path.join(dir,dir_model))
    #print("---------------------")
    model = tf.keras.models.load_model(os.path.join(dir_path,dir_model))

    # Check its architecture
    model.summary()
    app.run(debug=True) #indico que no necesito reiniciar para ver cambios
    #app.run() 