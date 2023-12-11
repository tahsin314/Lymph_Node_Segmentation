from losses.tversky import tversky_loss, focal_tversky
from losses.dice import dice_lossv2, dice_loss

def hybrid_loss(y_true, y_pred, alpha=1, beta=2):
    tversky_loss_value = focal_tversky(y_true, y_pred)
    dice_loss_value = dice_lossv2(y_true, y_pred)
    # print(tversky_loss_value.detach().cpu().numpy(), dice_loss_value.detach().cpu().numpy())
    return alpha*tversky_loss_value + beta*dice_loss_value
