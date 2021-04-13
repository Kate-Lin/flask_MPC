from flask_script import Manager
from flask_migrate import Migrate,MigrateCommand
from flask_MPC import app,db
from flask_MPC.models.voter import voter

migrate = Migrate(app,db)
manager = Manager(app)
manager.add_command('db',MigrateCommand)

if __name__ == '__main__':
    manager.run()