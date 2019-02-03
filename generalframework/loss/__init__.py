from .loss import CrossEntropyLoss2d, MSE_2D, PartialCrossEntropyLoss2d, JSD_2D, KL_Divergence_2D, Entropy_2D
import numpy as np

__all__ = ['get_loss_fn', 'enet_weighing']

LOSS = {'cross_entropy': CrossEntropyLoss2d,
        'mse_2d': MSE_2D,
        'partial_ce': PartialCrossEntropyLoss2d,
        'jsd': JSD_2D}


def get_loss_fn(name: str, **kwargs):
    try:
        return LOSS.get(name)(**kwargs)
    except Exception as e:
        raise ValueError('name error when inputting the loss name, with %s' % str(e))


def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:
        w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class:
        propensity_score = freq_class / total_pixels.
    References: https://arxiv.org/abs/1606.02147
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.
    """
    class_count = 0
    total = 0
    print("-> Calculate class weights....")
    from tqdm import tqdm
    for [img, label], _, _ in tqdm(dataloader):
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()
        mask = (flat_label>=0) & (flat_label<num_classes)
        flat_label = flat_label[mask]

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))
    print('the class weights done')

    return class_weights
