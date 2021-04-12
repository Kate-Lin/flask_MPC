from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config.from_object('flask_evoting.setting')
db = SQLAlchemy(app)

from flask_evoting.models import voter,bulletin,enc_vote,enc_sum
from flask_evoting.controller import home