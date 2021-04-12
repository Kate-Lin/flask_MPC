from flask_evoting import db


class voter(db.Model):
    __tablename__ = 'voter'
    ID = db.Column(db.INT,primary_key=True,autoincrement=True,comment='编号，用于拆分选票')
    voter_ID = db.Column(db.String(8),nullable=False,unique=True,comment='学号（随机生成）')
    check_num = db.Column(db.String(64),nullable=False,unique=True,comment='用于校验的sha256码')
    status = db.Column(db.INT,nullable=False)
    def __init__(self,voter_ID,check_num,status):
        self.voter_ID = voter_ID
        self.check_num = check_num
        self.status = status
    def __repr__(self):
        return '<User %r>' % self.voter_ID