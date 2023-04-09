import numpy as np
import copy
from act_func import activation,differential
from data import train_img,train_lab,valid_img,valid_lab
# 前向传播
def predict(img,parameters):
    hidden_in=np.dot(img,parameters[0]['w'])+parameters[0]['b']
    hidden_out=activation[0](hidden_in)
    l0_in=np.dot(hidden_out,parameters[1]['w'])+parameters[1]['b']
    l0_out=activation[1](l0_in)
    return l0_out

# 反向传播
def grad_parameters(img, lab, parameters, ll):
    ll = ll
    hidden_in = np.dot(img, parameters[0]['w']) + parameters[0]['b']
    hidden_out = activation[0](hidden_in)
    last_in = np.dot(hidden_out, parameters[1]['w']) + parameters[1]['b']
    last_out = activation[1](last_in)

    diff = onehot[lab] - last_out
    act1 = np.dot(differential[activation[1]](last_in), diff)

    grad_b1 = -2 * act1
    grad_w1 = -2 * (np.outer(hidden_out, act1) + ll * parameters[1]['w'])
    grad_b0 = -2 * differential[activation[0]](hidden_in) * np.dot(parameters[1]['w'], act1)
    # print(grad_b0.shape)
    grad_w0 = -2 * (np.outer(img, (differential[activation[0]](hidden_in) * np.dot(parameters[1]['w'], act1))) + ll *
                    parameters[0]['w'])

    return {'w1': grad_w1, 'b1': grad_b1, 'w0': grad_w0, 'b0': grad_b0}

# 训练
batch_size=100
def train_batch(current_batch,parameters,ll):
    grad_accu=grad_parameters(train_img[current_batch*batch_size+0],train_lab[current_batch*batch_size+0],parameters,ll)
    for img_i in range(1,batch_size):
        grad_tmp=grad_parameters(train_img[current_batch*batch_size+img_i],train_lab[current_batch*batch_size+img_i],parameters,ll)
        for key in grad_accu.keys():
            grad_accu[key]+=grad_tmp[key]
    for key in grad_accu.keys():
        grad_accu[key]/=batch_size
    return grad_accu

# 更新参数
def combine_parameters(parameters,grad,learn_rate):
    parameter_tmp=copy.deepcopy(parameters)
    parameter_tmp[0]['b']-=learn_rate*grad['b0']
    parameter_tmp[0]['w']-=learn_rate*grad['w0']
    parameter_tmp[1]['b']-=learn_rate*grad['b1']
    parameter_tmp[1]['w']-=learn_rate*grad['w1']
    return parameter_tmp

onehot=np.identity(10)


def sqr_loss(img,lab,parameters, ll):
    ll = ll
    y_pred=predict(img,parameters)
    y=onehot[lab]
    diff=y-y_pred
    return 0.5*np.dot(diff,diff)+ll*np.sum(np.square(parameters[0]['w']))+ll*np.sum(np.square(parameters[1]['w']))