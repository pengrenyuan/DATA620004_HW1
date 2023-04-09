from valid import test_accuracy
import pickle


if __name__ == '__main__':
    model_prameters_name = './Mnist_model.pkl'
    f = open(model_prameters_name, 'rb')
    param = pickle.load(f)
    # print(param)
    f.close

    accu = test_accuracy(param)
    print(f'加载模型在测试集上的测试准确率为{accu}')
    print('参数可视化：')
    print('第一层网络：','\n',param[0],'\n','第二层网络：','\n',param[1])