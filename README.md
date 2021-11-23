2021毕设--MPC
基于安全多方计算的隐私保护系统设计与实现
应用：
--大规模电子投票
--人工智能加密训练
# **《基于安全多方计算的隐私保护系统设计与实现》**
---
1. 项目位置：
   - 使用虚拟机镜像文件创建虚拟机，项目位于~/flask_MPC目录下.
   - [sudo su root]
   - cd ~/flask_MPC
2. 项目依赖：
   - 见~/flask_MPC/requirements.txt
   - 安装方法：pip install -r ~/flask_MPC/requirements.txt
3. 项目运行：
   - 在安装了上述依赖之后，
   - cd ~/flask_MPC
   - python runserver.py
   - 如果是要在架设的服务器上运行：
      - 首先需要开启nginx服务：/etc/init.d/nginx start
      - 根据本机IP修改nginx配置文件，位于：/etc/nginx/sites-available/default
      - 安装gunicorn：pip install gunicorn
      - 目录切换至flask_MPC文件下：cd ~/flask_MPC
      - 启动gunicorn，如：gunicorn -w 2 -b 127.0.0.1:5000 runserver:app --timeout=10000
      - 其中 -w -b --timeout都可自定义设置 --timeout可不要
4. mysql数据库配置检查：
   - mysql -u root -p
   - 密码为990207
   - 共有四张表，表的具体内容见论文的数据库介绍
   - 如果要测试投票功能，生成投票后的各人的验证码可通过以下方式查询：select * from voter;
5. 电子投票管理员的验证码：
   -c9a31a3f670b7f9973f2004ed383fc8c50a20c8d595556b8d8c266630234d8ee
