from keras import backend as K


def dice_coef(y_true, y_pred):
    """
    A simple dice metric over true and predicted tensors.
    We do not calculate dice axis-based, instead of this we calculate total dice
    for the whole batch. It is simpler, when the result is the same as after
    'fair' calculation of dice for each sample and averaging of it over the batch.
    dice = (2* |true ∩ pred|) / (|true|+|pred|)
    Lies in the interval [0, 1]
    """
    smooth = 1e-20  # smoothing for a case when true + pred = 0
    intersection = K.sum(y_true * y_pred)
    return (2 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def dice_coef_loss(y_true, y_pred):
    """
    Loss function with dice coefficient. The higher dice the better, hence loss
    may be just "-dice" or "1-dice" as here, for better aesthetics, as dice lies
    in the interval [0,1].
    """
    return 1 - dice_coef(y_true, y_pred)