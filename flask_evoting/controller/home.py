from flask_evoting.models.voter import voter
from flask_evoting.models.bulletin import bulletin
from flask_evoting.models.enc_vote import enc_vote
from flask_evoting.models.enc_sum import enc_sum
import numpy as np
from flask_evoting import app,db
from flask import request,render_template,flash,abort,url_for,redirect,session,Flask,g
from flask_evoting.controller.generate_voters import generate_voter_ID
from math import log2,floor
from torch import randperm
import heapq

@app.route('/')
def show_home():
    return render_template('index.html')

@app.route('/newvote')
def newvote():
    bullet = bulletin.query.all()
    if bullet == []:
        remain = None
    elif bullet[0].voter_num == bullet[0].T:
        remain = None
    else:
        remain = bullet[0].voter_num-bullet[0].T
    return render_template('new_vote.html',remain=remain)


@app.route('/vote')
def show_vote():
    created=bulletin.query.count()
    #当前没有投票发起，报错
    if created == 0:
        return 'ERROR! NO voting available!!'
    else:
        candidate = bulletin.query.first().candidate_num
        win = bulletin.query.first().win_num
        return render_template('vote.html',can_num = candidate+1,win=win)


@app.route('/create',methods=['POST'])
def create_vote():
    check_num = request.form['check_num']
    voter_num = int(request.form['voter_num'])
    candidate_num = int(request.form['candidate_num'])
    win_num = int(request.form['win_num'])
    if check_num != 'c9a31a3f670b7f9973f2004ed383fc8c50a20c8d595556b8d8c266630234d8ee':
        return "ERROR!! Invalid creator!!"
    else:
        db.reflect(app=app)
        for table_name in db.metadata.tables:
            db.get_engine().execute(f"truncate table {table_name}")
        # 创建随机用户学号
        generate_voter_ID(voter_num)
        # 创建投票公告板
        b = bulletin(voter_num=voter_num,candidate_num=candidate_num,win_num=win_num,k=floor(log2(voter_num))+1,z=0,T=0)
        db.session.add(b)
        db.session.commit()
        return "OK!!!"

@app.route('/process_result',methods=['POST'])
def vote():
    # 校验学号是否有权投票
    stu = voter.query.filter(voter.check_num==request.form['check_num']).all()
    if stu == []:
        return "ERROR!!! Invalid voter!!"
    elif stu[0].status == 1:
        return "ERROR!!! You have voted already!!"
    stu = stu[0]
    ID = stu.ID
    #print(ID)
    bullet = bulletin.query.first()
    k = bullet.k
    can_num = bullet.candidate_num
    voter_num = bullet.voter_num
    index = np.array(request.form.getlist('can'),dtype=int)
    #print(index)
    pi = 0
    for i in index:
        pi += 2**((can_num-i)*k)
    randnum = randperm(voter_num).numpy()
    randsum = np.sum(randnum)
    split = np.around(pi * randnum/randsum)
    #print(split)
    # 校准误差
    split[-1] += pi-np.sum(split)
    #投票人投票状态更改
    stu.status = 1
    bullet.T = bullet.T+1
    for i in range(bullet.voter_num):
        enc = enc_vote(x=ID-1,y=i,val=split[i])
        db.session.add(enc)
    if bullet.T == bullet.voter_num:
        bullet.z = 1
    db.session.commit()
    return "vote successfully!!!"

@app.route('/manage',methods=['POST'])
def process_manage():
    if request.form['check_num'] != 'c9a31a3f670b7f9973f2004ed383fc8c50a20c8d595556b8d8c266630234d8ee':
        return "ERROR!!INVALID creator!!"
    choice = request.form.getlist('manage')[0]

    if choice == 'auto':
        bullet = bulletin.query.first()
        stu = voter.query.filter(voter.status == 0).all()
        print(stu)
        for student in stu:
            random_vote(student,bullet)
        return "投票已完成"
    else:
        db.reflect(app=app)
        for table_name in db.metadata.tables:
            db.get_engine().execute(f"truncate table {table_name}")
        return "投票已重置，请返回首页."


def random_vote(student,bullet):
    ID = student.ID
    #随机出任意的选择，最多选两个的话：随机出0，1，2个选项
    k = bullet.k
    can_num = bullet.candidate_num
    voter_num = bullet.voter_num
    win = bullet.win_num
    index = randperm(can_num)[0:randperm(win+1)[0:1]].numpy()
    pi = 0
    for i in index:
        pi += 2 ** ((can_num - i - 1) * k)
    randnum = randperm(voter_num).numpy()
    randsum = np.sum(randnum)
    split = np.around(pi * randnum/randsum)
    #print(split)
    # 校准误差
    split[-1] += pi-np.sum(split)
    #投票人投票状态更改
    student.status = 1
    bullet.T = bullet.T+1
    if bullet.T == bullet.voter_num:
        bullet.z = 1
    for i in range(bullet.voter_num):
        enc = enc_vote(x=ID-1,y=i,val=split[i])
        db.session.add(enc)
    db.session.commit()

@app.route('/show_result')
def show_result():
    #每个计票人存储v'
    bullet = bulletin.query.first()
    if bullet.z == 0:
        return "投票尚未结束，请返回首页投票"
    db.reflect(app=app)
    db.get_engine().execute(f"truncate table enc_sum")
    for i in range(bullet.voter_num):
        voting = enc_vote.query.filter(enc_vote.y == i).all()
        sum = 0
        for v in voting:
            sum += v.val
        s = enc_sum(sender=i,val=sum)
        db.session.add(s)
    db.session.commit()
    sum = enc_sum.query.all()
    target = 0
    for s in sum:
        target += s.val
    k = bullet.k
    results = []
    for i in range(bullet.candidate_num):
        results.append(target & (2**k-1))
        target = target>>k
    results.reverse()
    print(results)
    return render_template('cal_sum.html',results=results)