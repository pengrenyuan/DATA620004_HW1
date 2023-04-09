from forward_bp import predict,sqr_loss
from data import valid_num,train_num,test_num,train_img,valid_lab,valid_img,test_img,test_lab,train_lab



# 指标
def valid_loss(parameters,ll):
    loss_accu=0
    for img_i in range(valid_num):
        loss_accu+=sqr_loss(valid_img[img_i],valid_lab[img_i],parameters,ll)
    # return loss_accu/(valid_num/10000)
    return loss_accu / valid_num


def valid_accuracy(parameters):
    correct=[predict(valid_img[img_i],parameters).argmax()==valid_lab[img_i] for img_i in range(valid_num)]
    return correct.count(True)/len(correct)


def train_loss(parameters,ll):
    loss_accu=0
    for img_i in range(train_num):
        loss_accu+=sqr_loss(train_img[img_i],train_lab[img_i],parameters,ll)
    # return loss_accu/(train_num/10000)
    return loss_accu / train_num

def train_accuracy(parameters):
    correct=[predict(train_img[img_i],parameters).argmax()==train_lab[img_i] for img_i in range(train_num)]
    return correct.count(True)/len(correct)


def test_accuracy(parameters):
    correct=[predict(test_img[img_i],parameters).argmax()==test_lab[img_i] for img_i in range(test_num)]
    return correct.count(True)/len(correct)