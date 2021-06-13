from flask import Flask, render_template

app = Flask(__name__)

#rutas
@app.route('/') #pagina principal
def home():
    return render_template('home.html')

#about
@app.route('/about') #pagina principal
def about():
    #return 'Sobre la página'
    return render_template('about.html')

#escuchar siempre
if __name__ == '__main__':
    app.run(debug=True) #indico que no necesito reiniciar para ver cambios
    #app.run() 