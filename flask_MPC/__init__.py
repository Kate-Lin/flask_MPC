from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__,template_folder='templates/',static_folder='static/')
app.config.from_object('flask_MPC.setting')
db = SQLAlchemy(app)

from flask_MPC.models import voter,bulletin,enc_vote,enc_sum
from flask_MPC.controller import home,evoting_manager,evoting_voter,AI_test