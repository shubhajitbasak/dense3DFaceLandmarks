from easydict import EasyDict as edict

config = edict()

config.task = 0

config.model_name = 'mobilenetv2_100' # Sparnet

config.dataset_name = 'wlpuv'
config.dataset_path = '/mnt/satan/data/300W_LP_UV'
# #    num_samples: 61225 #122450
config.batch_size = 128
config.shuffle = True
config.num_workers = 0
config.pin_memory = True
config.img_size = 256
config.posmap_size = 256
config.is_aug = True
config.min_blur_resize = 75
config.max_noise_var = 0.01
config.max_rot = 45
config.min_scale = 0.95
config.max_scale = 1.05
config.max_shift = 0.
config.num_verts = 520
config.uv_kpt_ind = 'data/uv_kpt_ind.txt'  # 2 x 68 get kpt
config.face_ind = 'data/face_ind.txt'  # get valid vertices in the pos map
config.triangles = 'data/triangles.txt'
config.filtered_indexs = 'data/vertices_520_sel_from_blender.txt'
config.filtered_68_kpt = 'data/vertices_68.txt'
config.filtered_kpt_500 = 'data/vertices_68_fil_520.txt'
config.resolution_inp = 256
config.resolution_op = 256
# config.keypoints = 520

config.network = "resnet_jmlr"
config.is_train = True
config.use_onenetwork = True
config.max_epochs = 40
config.lr = 0.01
config.export_path = ""
config.checkpoint_path = ""
config.load_checkpoint = -1
config.lr_policy = "step"
config.lr_decay_iters = 10

config.width_mult = 1.0

config.opt = 'sgd'
config.lr = 0.1  # when batch size is 512
config.momentum = 0.9
config.weight_decay = 5e-4
config.fc_mom = 0.9

config.warmup_epochs = 0
config.max_warmup_steps = 6000
config.num_epochs = 40

config.lr_func = None
config.lr_epochs = None

config.lossw_verts3d = 8.0
config.lossw_verts2d = 16.0
config.lossw_bone3d = 10.0
config.lossw_bone2d = 10.0
config.lossw_project = 10.0

# model

config.N_CLASS = 500
config.heatmap3d = False
config.kernel_size = 1
config.final_conv_kernel_size = 1
config.model_position_use_dw = False
config.output_hm_shape = (8, 8, 8)
config.imageNetNorm = False
config.pretrained = True
config.multiscale = False


# Evaluation

config.use_cam = True
config.eval_img_path = '/mnt/satan/data/AFLW2000-3D/AFLW2000'
config.is_dlib = False
config.is_mp = True