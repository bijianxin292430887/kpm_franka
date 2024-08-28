import os
import argparse
import time
import subprocess
import concurrent.futures
import numpy as np
import re
import sys
def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    for line in process.stderr:
        sys.stderr.write(line)
        sys.stderr.flush()
    process.wait()  # Wait for the process to complete
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")

time.sleep(0)
device = 'cuda:0'


# config = 'spoon_config_cotrain'
# agent = 'idm_latent_policy'
# agent_name='idm_latent_policy_resnet'
config = 'spoon_config'
agent = 'df_policy'
agent_name = 'df_policy_test'
command1 = (f'python run.py --config-name={config} agent={agent} agent_name={agent_name} device={device}')


# agent_name='idm_cotrain_single_koopman'
# command1 = (f'python run.py --config-name={config} agent={agent_name} agent_name={agent_name} device={device}')

# agent_name='idm_agent_baseline'
# command1 = (f'python run_robot_sim.py --config-name={config} agents=idm_baseline agent_name={agent_name} device={device}')

os.system(command1)