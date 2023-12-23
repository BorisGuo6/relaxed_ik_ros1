#!/bin/bash


for i in {0..71}
do
   echo "Obj Iter $i"
   # 假设你要拼接的命令是cmd，那么你可以这样做：
   padded_number=$(printf "%03d" "$i")
   cmd="python grasp_demo_3finger.py -o trimesh_$padded_number"
   eval $cmd
done