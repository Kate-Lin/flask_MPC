from flask_MPC.models.voter import voter
from flask_MPC.models.bulletin import bulletin
from flask_MPC.models.enc_vote import enc_vote
from flask_MPC.models.enc_sum import enc_sum
import numpy as np
from flask_MPC import app,db
from flask import request,render_template,flash,abort,url_for,redirect,session,Flask,g,jsonify
from flask_MPC.controller.generate_voters import generate_voter_ID
from math import log2,floor
from torch import randperm
import random
import json

@app.route('/evoting/newvote')
def newvote():
    bullet = bulletin.query.all()
    if bullet == []:
        remain = 0
    elif bullet[0].voter_num == bullet[0].T:
        remain = 0
    else:
        remain = bullet[0].voter_num-bullet[0].T
    return render_template('EVOTING_CREATE.html',remain=remain)

@app.route('/evoting/check_man_auth',methods=['POST'])
def check_man_auth():
    data = json.loads(request.form.get('data'))
    check=data['check']
    if check != 'c9a31a3f670b7f9973f2004ed383fc8c50a20c8d595556b8d8c266630234d8ee':
        return jsonify({'status':400,'message':'验证码错误'})
    else:
        return jsonify({'status':200,'message':'验证成功'})

@app.route('/evoting/check_auto_auth',methods=['POST'])
def check_auto_auth():
    data = json.loads(request.form.get('data'))
    check=data['check']
    bullet = bulletin.query.all()
    if bullet == []:
        return jsonify({'message':'当前无可用投票'})
    if check != 'c9a31a3f670b7f9973f2004ed383fc8c50a20c8d595556b8d8c266630234d8ee':
        return jsonify({'status':400,'message':'验证码错误'})
    else:
        bullet = bulletin.query.first()
        stu = voter.query.filter(voter.status == 0).all()
        print(stu)
        for student in stu:
            random_vote(student,bullet)
    return jsonify({'message':'随机选票已生成'})

@app.route('/evoting/check_set_auth',methods=['POST'])
def check_set_auth():
    data = json.loads(request.form.get('data'))
    check=data['check']
    if check != 'c9a31a3f670b7f9973f2004ed383fc8c50a20c8d595556b8d8c266630234d8ee':
        return jsonify({'message':'验证码错误'})
    else:
        db.reflect(app=app)
        for table_name in db.metadata.tables:
            db.get_engine().execute(f"truncate table {table_name}")
    return jsonify({'message':'投票已重置'})

@app.route('/evoting/create',methods=['POST'])
def create_vote():
    check_num = request.form['check_num']
    voter_num = int(request.form['voter_num'])
    candidate_num = int(request.form['candidate_num'])
    win_num = int(request.form['win_num'])
    if check_num != 'c9a31a3f670b7f9973f2004ed383fc8c50a20c8d595556b8d8c266630234d8ee':
        return render_template('error.html',message='新建投票失败')
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
        return render_template('success.html',message='新建投票')

@app.route('/evoting/manage',methods=['POST'])
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
    index = randperm(can_num)[0:win].numpy()
    print(ID, index)
    pi = 0
    for i in index:
        pi += 2 ** ((can_num - i - 1) * k)
    split = random.sample(range(-10000,10000), voter_num-1)
    print(split)
    # 校准误差
    split.append(pi-np.sum(np.array(split)))
    print(split)
    #投票人投票状态更改
    student.status = 1
    bullet.T = bullet.T+1
    if bullet.T == bullet.voter_num:
        bullet.z = 1
    for i in range(bullet.voter_num):
        enc = enc_vote(x=ID-1,y=i,val=split[i])
        db.session.add(enc)
    db.session.commit()