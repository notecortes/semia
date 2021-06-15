from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Email, Length

class Fotos(FlaskForm):
    nombre = StringField("Nombre", validators =[
        DataRequired(),
        Length(max=50, min=2)
    ])