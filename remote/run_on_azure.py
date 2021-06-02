"""
Based on
https://github.com/rail-berkeley/doodad/blob/master/testing/remote/test_azure_with_mounts.py

Instructions:
1) Set up testing/config.py (copy from config.py.example and fill in the fields)
2) Run this script
3) Look inside your AZ_CONTAINER and you should see results in test_azure_with_mounts/azure_script_output/output.out
"""
import os

import doodad
from doodad.utils import TESTING_DIR

AZ_SUB_ID = os.environ['AZURE_SUBSCRIPTION_ID']
AZ_CLIENT_ID = os.environ['AZURE_CLIENT_ID']
AZ_TENANT_ID = os.environ['AZURE_TENANT_ID']
AZ_SECRET = os.environ['AZURE_CLIENT_SECRET']
AZ_CONTAINER = os.environ['AZURE_STORAGE_CONTAINER']
AZ_CONN_STR = os.environ['AZURE_STORAGE_CONNECTION_STRING']


def run():
    launcher = doodad.AzureMode(
        azure_subscription_id=AZ_SUB_ID,
        azure_storage_connection_str=AZ_CONN_STR,
        azure_client_id=AZ_CLIENT_ID,
        azure_authentication_key=AZ_SECRET,
        azure_tenant_id=AZ_TENANT_ID,
        azure_storage_container=AZ_CONTAINER,
        log_path='jax-rl',
        region='eastus',
        instance_type='Standard_DS1_v2',
        # To run on GPU comment instance_type and uncomment lines below.
        # use_gpu=True,
        # gpu_model='nvidia-tesla-t4',
        # num_gpu=1
    )

    az_mount = doodad.MountAzure(
        'azure_script_output',
        mount_point='/output',
    )
    local_mount = doodad.MountLocal(local_dir=TESTING_DIR,
                                    mount_point='/data',
                                    output=False)
    code_mount1 = doodad.MountLocal(local_dir='/home/kostrikov/GitHub/jax-rl/',
                                    mount_point='/code/jax-rl',
                                    pythonpath=True)
    mounts = [local_mount, az_mount, code_mount1]

    doodad.run_command(
        command=
        'python -u /code/jax-rl/train.py --env_name=HalfCheetah-v2 --max_steps=10000 --config=/code/jax-rl/configs/sac_default.py  --save_dir=/output/tmp/ > /output/output.out 2> /output/output.err',
        mode=launcher,
        mounts=mounts,
        verbose=True,
        docker_image="ikostrikov/jax-rl:latest",
    )


if __name__ == '__main__':
    run()
