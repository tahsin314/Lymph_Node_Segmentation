
''' You only need to change the config_parmas dictionary'''
config_params = dict(
    data_dir = "/shared/rail_lab/thyroid_cartilage/thyroid_slices_2_npz/", #"../DATA/Clouds/38-Cloud_training"
    model_dir = 'model_dir',
    model_name = 'SNet',
    n_fold = 5,
    fold = 1,
    sz = 512,
    num_slices = 2,
    threshold = 0.25,
    dataset = 'Thyroid Cartilage Segmentation',
    lr = 1.25e-4,
    eps = 1e-5,
    weight_decay = 1e-5,
    n_epochs = 60,
    bs = 16,
    gradient_accumulation_steps = 2,
    SEED = 2023,
    sampling_mode = None, #upsampling
    pretrained = False,
    mixed_precision = False
    )

