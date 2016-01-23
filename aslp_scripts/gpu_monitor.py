#!/usr/bin/env python
import paramiko
import os
import sys

# Query node in machine.conf and get gpu usage
config_file = ''
if os.path.exists('machine.conf'):
    # When in cur dir
    config_file = 'machine.conf'
elif os.path.exists('aslp_scripts/machine.conf'):
    # config file in current aslp_scripts dir
    config_file = 'aslp_scripts/machine.conf'
else:
    print 'Error, no such file: machine.conf'
    sys.exit(1) 

with open(config_file) as fid:
    for line in fid.readlines():
        if line.strip().startswith('#'):  continue
        fileds = line.split()
        if len(fileds) < 4: continue
        machine, ip, user, password = fileds[0], fileds[1], fileds[2], fileds[3]
        #print machine, ip, user, password
        try:
            print 'Machine: %s %s' % (machine, ip)
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, 22, user, password)
            stdin, stdout, stderr = ssh.exec_command('nvidia-smi')
            for msg in stdout.readlines():
                print msg[:-1]
            print ''
            ssh.close()
        except:
            continue
