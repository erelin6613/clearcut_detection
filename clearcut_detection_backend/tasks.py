import os
import threading

from invoke import task


@task
def run(ctx):
    init_db(ctx, recreate_db=False)
    # collect_static_element(ctx)
    # thread_cron = threading.Thread(target=devcron, args=(ctx,))
    # thread_cron.start()
    # ctx.run('python update.py')

@task
def devcron(ctx):
    ctx.run('python devcron.py cron_tab_prod')


@task
def collect_static_element(ctx):
    ctx.run('python manage.py collectstatic --noinput')
    ctx.run('python manage.py compilemessages')


@task
def init_db(ctx, recreate_db=False):
    wait_port_is_open(os.getenv('DB_HOST', 'db'), 5432)
    if recreate_db:
        pass
        # ctx.run('python manage.py dbshell < clear.sql')
        # ctx.run('python manage.py dbshell < db.dump2404191230')

    ctx.run('python manage.py makemigrations clearcuts')
    ctx.run('python manage.py migrate')


def wait_port_is_open(host, port):
    import socket
    import time
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return
        except socket.gaierror:
            pass
        time.sleep(1)
