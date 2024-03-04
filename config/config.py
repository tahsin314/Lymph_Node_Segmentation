
''' You only need to change the config_params dictionary'''
config_params = dict(
    data_dir = "../../data/lymph_node/ct_221_0_npz", #"../DATA/Clouds/38-Cloud_training"
    model_dir = 'model_dir_static_lr',
    model_name = 'SNet',
    n_fold = 5,
    fold = 3,
    device_id = 3,
    sz = 384,
    num_slices = 0,
    threshold = 0.5,
    dataset = 'LN Segmentation',
    lr = 5e-5,
    eps = 1e-5,
    weight_decay = 1e-5,
    n_epochs = 100,
    bs = 16,
    gradient_accumulation_steps = 1,
    SEED = 2023,
    sampling_mode = None, #upsampling
    pretrained = False,
    mixed_precision = False
    )