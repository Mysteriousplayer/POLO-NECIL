import sys
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,5,6,7"
import torch
import torch.utils.data

import argparse
import logging
import copy

from NECIL_subimagenet import polo
from ResNet_imagenet import resnet18_ImageNet

from myNetwork_imagenet import network, network_cosine
from data_manager_imagenet import *

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser(description='Prototype Augmentation and Self-Supervision for Incremental Learning')
parser.add_argument('--epochs', default=151, type=int, help='Total number of epochs to run')
parser.add_argument('--decay_epochs', default=60, type=int, help='Total number of epochs to reduce lr')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='sub100', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=50, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=5, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--incre_lr', default=0.001, type=float, help='incremental learning rate')
parser.add_argument('--protoAug_weight', default=10.0, type=float, help='protoAug loss weight')
parser.add_argument('--kd_weight', default=10.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--temp', default=1.0, type=float, help='trianing time temperature')
parser.add_argument('--gpu', default='9', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')
parser.add_argument('--log_path', default='', type=str, help='save logs')
parser.add_argument('--drift_weight', default=1.0, type=float, help='drift weight')
parser.add_argument('--file_name', default='', type=str, help='file name')
parser.add_argument('--density', default=1, type=int, help='density-based prototype')
parser.add_argument('--alpha', default=20, type=float, help='hyper parameter')
parser.add_argument('--beta', default=0.1, type=float, help='hyper parameter')
parser.add_argument('--gamma', default=10, type=float, help='hyper parameter')
parser.add_argument('--alpha_2', default=10, type=float, help='hyper parameter')
args = parser.parse_args()


def main():
    ssl = 4
    sys.stdout = Logger(args.log_path)
    print(args)
    prototype = None

    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")

    task_size = int((args.total_nc - args.fg_nc) / args.task_num)  # number of classes in each incremental step
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(task_size) + '_' + args.file_name
    feature_extractor = resnet18_ImageNet()
    data_manager = DataManager()
    model = polo(args, file_name, feature_extractor, task_size, device)
    class_set = list(range(args.total_nc))

    for i in range(args.task_num+1):
        if i == 0:
            old_class = 0
        else:
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])
        model.beforeTrain(i, data_manager, class_set)
        model.train(i, old_class=old_class)
        prototype = model.afterTrain()


    print("############# Test for each Task #############")
    test_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    class_set = list(range(100))
    acc_all = []
    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = network_cosine(class_index * ssl, feature_extractor)
        state_dict = torch.load(filename)
        model.load_state_dict(state_dict)
        model.to(device)

        model.eval()
        acc_up2now = []
        for i in range(current_task + 1):
            if i == 0:
                classes = class_set[:args.fg_nc]
            else:
                classes = class_set[(args.fg_nc + (i - 1) * task_size):(args.fg_nc + i * task_size)]
            testfolder = data_manager.get_dataset(test_transform, index=classes, train=False)
            test_loader = torch.utils.data.DataLoader(
                testfolder, batch_size=100,
                shuffle=False,
                drop_last=True, num_workers=4)
            correct, total = 0.0, 0.0
            for setp, data in enumerate(test_loader):
                imgs, labels = data
                imgs, labels = imgs.cuda(), labels.cuda()
                with torch.no_grad():
                    outputs = model(imgs)
                    outputs = outputs[:, :ssl * class_index]
                outputs = outputs[:, ::ssl]
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)
            accuracy = correct.item() / total
            acc_up2now.append(accuracy)
        if current_task < args.task_num:
            acc_up2now.extend((args.task_num - current_task) * [0])
        acc_all.append(acc_up2now)
        print(acc_up2now)
    print(acc_all)

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
    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        # model = torch.load(filename)
        model = network_cosine(class_index * ssl, feature_extractor)
        state_dict = torch.load(filename)
        model.load_state_dict(state_dict)
        model.to(device)

        model.eval()
        classes = class_set[:(args.fg_nc + current_task * task_size)]
        testfolder = data_manager.get_dataset(test_transform, index=classes, train=False)
        test_loader = torch.utils.data.DataLoader(
            testfolder, batch_size=100,
            shuffle=False,
            drop_last=False, num_workers=4)
        correct, total = 0.0, 0.0

        for setp, data in enumerate(test_loader):
            imgs, labels = data
            imgs, labels = imgs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = model(imgs)
                outputs = outputs[:, :ssl * class_index]
                outputs = outputs[:, ::ssl]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        print('incremental acc: ')
        print(accuracy)
        avg_acc.append(accuracy)
    print('average CNN acc:', sum(avg_acc) / len(avg_acc))

if __name__ == "__main__":
    main()