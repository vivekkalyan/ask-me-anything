import math
import time

def batch_acc(predicted, true):
    '''
    Calculate the accuracy of batch 
    According to paper, accuracy is calculated over the 10 choose-9 subsets
    Because the discarded answer is either the predicted answer or not
    - for it to be same -> 1 less same answer ((same) number of cases)
    - for it to be not same -> same-answer number stays constant ((10 - same) number of cases)
    => acc = ((10 - same) * min(same / 3, 1) + same * min((same - 1) / 3)) / 10
    if same == 0:
        acc = 0
    elif same >= 4:
        acc = 1
    else:
        we will have (same - 1) / 3 < same  / 3 <= 1
        => min(same / 3, 1) = same and min((same - 1) / 3, 1) = (same - 1) / 3
        => acc = ((10 - same) * same + same * (same - 1)) / 3 / 10
        => acc = same * (10 - same + same - 1) / 30 = same * 0.3

    ==> for all cases, we have: acc = min(same * 0.3, 1)
    '''
    _, predex = predicted.max(dim=1, keepdim=True)
    same = true.gather(dim=1, index=predex)

    return (same * 0.3).clamp(max=1)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
