from flask_MPC import db

class enc_sum(db.Model):
    __tablename__ = 'enc_sum'
    ID = db.Column(db.INT,primary_key=True,autoincrement=True)
    sender = db.Column(db.INT,unique=True,nullable=False)
    val = db.Column(db.BIGINT,nullable=False)