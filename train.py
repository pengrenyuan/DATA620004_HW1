import matplotlib.pyplot as plt
from para_search import hyperparams
from param_init import init_parameters
from forward_bp import batch_size,train_batch,combine_parameters
from data import train_num
from valid import train_loss,train_accuracy,valid_loss,valid_accuracy
import pickle


if __name__ == '__main__':
    hyper_accu = [0]
    for ii in range(3):
        print(f'开始第{ii+1}组超参数训练及验证')
        ll = hyperparams['L2'][ii]
        hidden_dim = hyperparams['hidden_dims'][ii]
        lr = hyperparams['lr'][ii]
        dimensions = [28 * 28, hidden_dim, 10]  # 各层的维度
        # 初始化参数
        parameters = init_parameters(dimensions)
        current_epoch = 0
        train_loss_list = []
        valid_loss_list = []
        train_accu_list = []
        valid_accu_list = []
        # 训练
        learn_rate = lr
        # learn_rate=1
        epoch_num = 30
        for epoch_ in range(epoch_num):
            for i in range(train_num // batch_size):
                grad_tmp = train_batch(i, parameters,ll)
                parameters = combine_parameters(parameters, grad_tmp, learn_rate)
            current_epoch += 1
            train_loss_list.append(train_loss(parameters,ll))
            train_accu_list.append(train_accuracy(parameters))
            valid_loss_list.append(valid_loss(parameters,ll))
            valid_accu_list.append(valid_accuracy(parameters))
            print(f'epoch:{current_epoch}/{epoch_num}  train/valid_loss:{train_loss_list[-1]}  {valid_loss_list[-1]}'
              f'  train/valid_accu:{train_accu_list[-1]}  {valid_accu_list[-1]}')
        # 验证集准确率
        accu = valid_accuracy(parameters)
        # 损失准确率可视化
        fig = plt.figure(figsize=(8,2),dpi=90)
        ax1 = fig.add_subplot(1,2,1)
        plt.plot(valid_accu_list, c='g', label='validation acc')
        plt.plot(train_accu_list, c='b', label='train acc')
        plt.legend()

        ax1 = fig.add_subplot(1, 2, 2)
        plt.plot(valid_loss_list, c='g', label='validation loss')
        plt.plot(train_loss_list, c='b', label='train loss')
        plt.legend()
        plt.savefig(f'./{ii}_loss_accu')


        # 保存效果最好的模型
        if accu > max(hyper_accu):
            model_prameters_name = './Mnist_model.pkl'
            f = open(model_prameters_name, 'wb')
            pickle.dump(parameters, f)
            f.close()
        hyper_accu.append(accu)
    print(hyper_accu)