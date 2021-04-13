from flask_MPC import db

class bulletin(db.Model):
    __tablename__ = 'bulletin'
    ID = db.Column(db.INT,primary_key=True,autoincrement=True)
    voter_num = db.Column(db.BIGINT,nullable=False)
    candidate_num = db.Column(db.BIGINT,nullable=False)
    win_num = db.Column(db.INT,nullable=False)
    k = db.Column(db.INT,nullable=False)
    z = db.Column(db.INT,nullable=False)
    T = db.Column(db.BIGINT,nullable=False)
