import time

def log(txt, sout=True):
    txt = '-- %s -- %s' % (time.asctime(), txt)
    if sout:
        print(txt)
