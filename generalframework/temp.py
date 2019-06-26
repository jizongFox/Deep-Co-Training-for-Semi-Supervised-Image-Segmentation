#
#
#   file description
#
#
__author__ = "Jizong Peng"
from deepclustering.postprocessing._viewer import  multi_slice_viewer
import matplotlib.pyplot as plt
multi_slice_viewer(img_2.squeeze(1).cpu(),[gt_2.squeeze(1).cpu(), segmentators[0].predict(img_2, logit=False).detach().max(1)[1].cpu(),
                                           segmentators[1].predict(img_2, logit=False).max(1)[1].cpu()])
plt.show()
multi_slice_viewer(img_adv[0].unsqueeze(0).squeeze(1).cpu(),[gt_2.squeeze(1).cpu(), segmentators[0].predict(img_adv, logit=False).detach().max(1)[1].cpu(),
                                                             segmentators[1].predict(img_adv, logit=False).max(1)[1].cpu()])
plt.show()

