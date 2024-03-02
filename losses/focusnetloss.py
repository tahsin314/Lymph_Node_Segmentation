from .tversky import tversky_loss
from .structure_loss import total_structure_loss_focusnet

def FocusNetLoss(mask, preds):
    outputs, lateral_map_2, lateral_map_1 = preds
    t_l = tversky_loss(mask, outputs)
    s_l = total_structure_loss_focusnet(mask, (lateral_map_2, lateral_map_1))
    return t_l + s_l
    