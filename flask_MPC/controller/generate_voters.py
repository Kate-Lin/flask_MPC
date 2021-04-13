import random
import hashlib
from flask_MPC.models.voter import voter
from flask_MPC import db,app

def generate_voter_ID(n):
    db.reflect(app=app)
    db.get_engine().execute(f"truncate table voter")
    k = 17120001
    randarray = random.sample(list(range(k,k+n*8)),n)
    s = hashlib.sha256()
    l = []
    for i in range(len(randarray)):
        s.update(str(randarray[i]).encode())
        l.append(s.hexdigest())
    return save_voter(randarray,l)

def save_voter(r,l):
    for i in range(len(l)):
        v = voter(voter_ID=r[i],check_num=l[i],status=0)
        db.session.add(v)
    db.session.commit()
