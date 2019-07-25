import os
OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', './outputs')
path = os.path.join(OUTPUTS_DIR, 'ok.txt')
f= open(path,"w+")
for i in range(10):
    f.write("This is line %d\r\n" % (i+1))
f.close()
