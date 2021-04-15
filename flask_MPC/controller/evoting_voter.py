from flask_MPC.models.voter import voter
from flask_MPC.models.bulletin import bulletin
from flask_MPC.models.enc_vote import enc_vote
from flask_MPC.models.enc_sum import enc_sum
import numpy as np
from flask_MPC import app,db
from flask import request,render_template,flash,abort,url_for,redirect,session,Flask,g
from flask_MPC.controller.generate_voters import generate_voter_ID
from math import log2,floor
from torch import randperm
import random

@app.route('/evoting/vote')
def show_vote():
    created=bulletin.query.count()
    #当前没有投票发起，报错
    if created == 0:
        return render_template('error.html',message='当前无可操作投票')
    else:
        candidate = bulletin.query.first().candidate_num
        win = bulletin.query.first().win_num
        return render_template('EVOTING_VOTE.html',voter_num=bulletin.query.first().voter_num,can_num = candidate+1,win=win)


@app.route('/evoting/show_result')
def show_result():
    #每个计票人存储v'
    bullet = bulletin.query.first()
    n = bullet.voter_num
    if bullet.z == 0:
        return "尚未结束，请返回首页投票"
    manual = n*n-enc_vote.query.count()
    print(manual)
    print(voter.query.filter(voter.finished==1).count())
    if voter.query.filter(voter.finished==1).count() != manual and voter.query.filter(voter.finished==1).count() != n:
        return "手动提交计票尚未结束，请等待。"
    auto = voter.query.filter(voter.finished==0).all()
    for a in auto:
        ID = a.ID-1
        remain_vote = 0
        enc_v = enc_vote.query.filter(enc_vote.y == ID).all()
        for v in enc_v:
            remain_vote += v.val
        a.finished = 1
        enc_s = enc_sum(sender=ID, val=remain_vote)
        db.session.add(enc_s)
    db.session.commit()
    """
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
    """
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


@app.route('/evoting/process_result',methods=['POST'])
def vote():
    # 校验学号是否有权投票
    name = request.form['check_num']
    vote_part = request.form['vote_part'].split(',')
    print(name)
    print(vote_part)
    stu = voter.query.filter(voter.check_num == name).all()
    if stu == []:
        return render_template('error.html',message='验证码有误，投票失败')
    elif stu[0].status == 1:
        return render_template('error.html', message='请勿重复投票')
    stu = stu[0]
    ID = stu.ID
    # print(ID)
    bullet = bulletin.query.first()
    k = bullet.k
    can_num = bullet.candidate_num
    voter_num = bullet.voter_num
    stu.status = 1
    bullet.T = bullet.T + 1
    ind = list(range(bullet.voter_num))
    ind.remove(ID-1)
    for i in range(len(vote_part)):
        enc = enc_vote(x=ID - 1, y=ind[i], val=int(vote_part[i]))
        db.session.add(enc)
    """
    for i in range(bullet.voter_num):
        if i != ID-1:
            enc = enc_vote(x=ID - 1, y=i, val=int(vote_part[i-1]))
            db.session.add(enc)
    """
    if bullet.T == bullet.voter_num:
        bullet.z = 1
    db.session.commit()
    return render_template('success.html',message='投票成功')

@app.route('/check_before_add')
def check_before_add():
    bullet = bulletin.query.first()
    if bullet.z == 0:
        return "投票尚未结束，请返回首页"
    return render_template('check_before_add.html')

@app.route('/evoting/local_check',methods=['POST'])
def local_check():
    check_num = request.form['check_num']
    stu = voter.query.filter(voter.check_num == check_num).all()
    if stu == []:
        return redirect(url_for('error'))
    #查找出该用户求和需要的数据，读取数据库
    ID = stu[0].ID-1
    remain_vote = 0
    enc_v = enc_vote.query.filter(enc_vote.y==ID).all()
    for v in enc_v:
        remain_vote += v.val
    return render_template('local_check.html',remain=remain_vote,user=ID)

@app.route('/evoting/save_sum',methods=['POST'])
def save_sum():
    total = int(request.form['total'])
    ID = int(request.form['name'])
    print('total sum:',total)
    print('voter_ID',ID)
    stu = voter.query.filter(voter.ID==ID+1).first()
    stu.finished = 1
    enc_s = enc_sum(sender=ID,val=total)
    db.session.add(enc_s)
    db.session.commit()
    return '本地计票完成'
