import os, time
import datetime

yaml_dir = "/home/madcreeper/rangedik_project/src/relaxed_ik_ros1/relaxed_ik_core/configs/grasp_yaml"


with open('time_log.txt', 'w') as f:
    f.write(f"{datetime.datetime.now()}\n")

for yaml_name in os.listdir(yaml_dir):
    if yaml_name[-5:] == '.yaml':
        yaml_name = yaml_name[:-5]
        print(yaml_name)


        os.system(f"python grasp_demo_3finger.py -o {yaml_name}")

        
        
        