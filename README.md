# Dense 3D Face Landmarks

Repository for the paper - [A lightweight 3D dense facial landmark estimation model from position map data](https://arxiv.org/pdf/2308.15170)

Prepare Data - 

Follow the instruction in [face3d](https://github.com/yfeng95/face3d.git) page to generate the UV map data from the [300W-LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) dataset.  

Make sure to install mesh_core_cython in the local python environment by running - 

```shell
# directory - face3d/mesh/cython
pip install cython 
python3 setup.py build_ext --inplace 
python3 setup.py install
```

* Point the BFM.mat and BFM_UV.mat in generate_posmap_300WLP.py
* Update the input_path and output_path in generate_posmap_300WLP.py

Training -

run - train_mobilnet.py

Inference - 

run - inference.py

Pretrained Checkpoint - 

https://drive.google.com/file/d/1pfUZRMzLh8m53RI3mOqbjDfJGeOZHE_i/view?usp=sharing

Check the Configs/config.py for the configuration details.

Sample Results - 

![Results.png](Results/Results.png)

### Citation

If you use this code, please consider citing:

```
@article{basak2023lightweight,
  title={A lightweight 3D dense facial landmark estimation model from position map data},
  author={Basak, Shubhajit and Mangapuram, Sathish and Costache, Gabriel and McDonnell, Rachel and Schukat, Michael},
  journal={arXiv preprint arXiv:2308.15170},
  year={2023}
}
```



