from .loss import CrossEntropyLoss2d, MSE_2D,PartialCrossEntropyLoss2d,JSD_2D,KL_Divergence_2D,Entropy_2D

LOSS = {'cross_entropy': CrossEntropyLoss2d,
        'mse_2d': MSE_2D,
        'partial_ce':PartialCrossEntropyLoss2d,
        'jsd':JSD_2D}


def get_loss_fn(name: str, **kwargs):
    try:
        return LOSS.get(name)(**kwargs)
    except Exception as e:
        raise ValueError('name error when inputting the loss name, with %s'%str(e))
