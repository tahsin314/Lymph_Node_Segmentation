from config.config import config_params
from models.mit.mit_PLD_b4 import mit_PLD_b4
from models.unet import U_Net
from models.mit.mit_PLD_b2 import mit_PLD_b2
from models.CaraNet.caranet import caranet
from models.FCBFormer.FCBFormer import FCBFormer
from models.DUCKNet.DUCKNet import DuckNet
from models.Focus_Net.s_net import s_net
from models.Focus_Net.s_net import s_net

model_params = dict(

    UNet = U_Net(2*config_params['num_slices'] + 1, 1),
    ssformer_S = mit_PLD_b2(class_num=1, in_chans=2*config_params['num_slices'] + 1),
    ssformer_L = mit_PLD_b4(class_num=1, in_chans=2*config_params['num_slices'] + 1),
    CaraNet = caranet(in_chans=2*config_params['num_slices'] + 1),
    FCBFormer = FCBFormer(size=config_params['sz']),
    DUCKNet = DuckNet(in_chans=2*config_params['num_slices'] + 1, starting_filters=11),
    SNet = s_net(channel=2*config_params['num_slices'] + 1, num_classes=1),
    )