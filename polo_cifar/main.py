import sys
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
import torch
import torch.utils.data
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
import argparse
import numpy as np
import copy
from POLO import polo
from ResNet import resnet18_cbam
from iCIFAR100 import iCIFAR100


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

parser = argparse.ArgumentParser(description='Prototype Augmentation and Self-Supervision for Incremental Learning')
parser.add_argument('--epochs', default=151, type=int, help='Total number of epochs to run')
parser.add_argument('--decay_epochs', default=60, type=int, help='Total number of epochs to reduce lr')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='cifar100', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=50, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=5, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--incre_lr', default=0.001, type=float, help='incremental learning rate')
parser.add_argument('--protoAug_weight', default=10.0, type=float, help='protoAug loss weight')
parser.add_argument('--kd_weight', default=10.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--cls_weight', default=1.0, type=float, help='semantic loss weight')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')
parser.add_argument('--log_path', default='', type=str, help='save logs')
parser.add_argument('--drift_weight', default=1.0, type=float, help='drift weight')
parser.add_argument('--file_name', default='', type=str, help='file name')
parser.add_argument('--density', default=1, type=int, help='density based prototype')
parser.add_argument('--alpha', default=20, type=float, help='hyper parameter')
parser.add_argument('--beta', default=0.1, type=float, help='hyper parameter')
parser.add_argument('--gamma', default=10, type=float, help='hyper parameter')
parser.add_argument('--alpha_2', default=10, type=float, help='hyper parameter')
args = parser.parse_args()

def map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def setup_data(test_targets, shuffle, seed):
    order = [i for i in range(len(np.unique(test_targets)))]
    if shuffle:
        np.random.seed(seed)
        order = np.random.permutation(len(order)).tolist()
    else:
        order = range(len(order))
    class_order = order
    print(100 * '#')
    print(class_order)
    return map_new_class_index(test_targets, class_order)

def main():
    sys.stdout = Logger(args.log_path)
    print(args)
    prototype = None
    d_ids = [0]
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)  # number of classes in each incremental step
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(task_size) + '_' + args.file_name
    feature_extractor = resnet18_cbam()

    model = polo(args, file_name, feature_extractor, task_size, device)
    class_set = list(range(args.total_nc))
    model.setup_data(shuffle=True, seed=1993)

    d = {}
    for i in range(args.task_num+1):
        if i == 0:
            old_class = 0
        else:
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])
        model.beforeTrain(i)
        model.train(i, old_class=old_class)
        prototype = model.afterTrain()
        d['proto{0}'.format(i)] = copy.deepcopy(prototype)


    ####### Test ######
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    print("############# Test for each Task #############")
    test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)
    test_dataset.targets = setup_data(test_dataset.targets, shuffle=True, seed=1993)
    acc_all = []
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.eval()
        acc_up2now = []
        for i in range(current_task+1):
            if i == 0:
                classes = [0, args.fg_nc]
            else:
                classes = [args.fg_nc + (i - 1) * task_size, args.fg_nc + i * task_size]
            test_dataset.getTestData_up2now(classes)
            test_loader = DataLoader(dataset=test_dataset,
                                     shuffle=True,
                                     batch_size=args.batch_size)
            correct, total = 0.0, 0.0
            for setp, (indexs, imgs, labels) in enumerate(test_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(imgs)

                outputs = outputs[:, ::4]
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)

            accuracy = correct.item() / total
            acc_up2now.append(accuracy)
        if current_task < args.task_num:
            acc_up2now.extend((args.task_num-current_task)*[0])
        acc_all.append(acc_up2now)
        print('CNN:', acc_up2now)

    print('CNN:', acc_all)

    a = np.array(acc_all)
    result = []
    for i in range(args.task_num + 1):
        if i == 0:
            result.append(0)
        else:
            res = 0
            for j in range(i + 1):
                res += (np.max(a[:, j]) - a[i][j])
            res = res / i
            result.append(100 * res)
    print(50 * '#')
    print('Forgetting result:')
    print(result)

    print("############# Test for up2now Task #############")
    avg_acc = []
    test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)
    test_dataset.targets = setup_data(test_dataset.targets, shuffle=True, seed=1993)
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.to(device)
        model.eval()

        classes = [0, args.fg_nc + current_task * task_size]
        test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=test_dataset,
                                 shuffle=True,
                                 batch_size=args.batch_size)
        correct, total, correct_nme = 0.0, 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(imgs)

            outputs = outputs[:, ::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        print('CNN:', accuracy)
        avg_acc.append(accuracy)

    print('average CNN acc:', sum(avg_acc)/len(avg_acc))

if __name__ == "__main__":
    main()