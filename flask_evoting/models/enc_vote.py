from flask_evoting import db

class enc_vote(db.Model):
    __tablename__ = 'enc_vote'
    ID = db.Column(db.INT,primary_key=True,autoincrement=True)
    x = db.Column(db.INT,nullable=False)
    y = db.Column(db.INT,nullable=False)
    val = db.Column(db.BIGINT,nullable=False)
