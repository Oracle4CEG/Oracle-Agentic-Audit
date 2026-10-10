"""Authenticated research-server commands; credentials never enter logs or argv."""
import argparse
import pexpect
from dotenv import dotenv_values


def run(command, env_file, timeout=3600):
    values=dotenv_values(env_file)
    target=values['sun-lab-username']+'@'+values['sun-lab-ip']
    child=pexpect.spawn('ssh',['-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=15',target,command],
                        encoding='utf-8',timeout=20)
    state=child.expect(['[Pp]assword:',pexpect.EOF])
    if state==0:
        child.sendline(values['sun-lab-pwd'])
        child.expect(pexpect.EOF,timeout=timeout)
    output=child.before.replace('\r\n','\n').lstrip()
    child.close()
    if child.exitstatus!=0:raise RuntimeError(f'Remote command exit {child.exitstatus}: {output}')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--env-file',required=True);p.add_argument('--command',required=True)
    args=p.parse_args();print(run(args.command,args.env_file),flush=True)
