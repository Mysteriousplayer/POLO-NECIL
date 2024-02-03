import torch
from myNetwork_tiny import network, network_cosine
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from data_manager_tiny import *
import copy

class polo:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.ssl = 4
        self.rate = 8
        self.file_name = file_name
        self.args = args
        self.epochs = args.epochs
        self.decay_epochs = args.decay_epochs
        self.learning_rate = args.learning_rate
        self.incre_lr = args.incre_lr
        self.model = network_cosine(args.fg_nc * self.ssl, feature_extractor)
        self.feature_extractor = feature_extractor

        self.radius = []
        self.prototype = None
        self.class_label = None
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None

        self.size = 64
        self.data_manager = DataManager()
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.train_loader = None
        self.test_loader = None
        self.cov_list = None
        self.drift_weight = args.drift_weight
        self.d_ids = [0, 1, 2, 3]

    def beforeTrain(self, current_task):
        self.model.eval()
        class_set = list(range(200))
        if current_task == 0:
            classes = class_set[:self.numclass]
        else:
            classes = class_set[self.numclass-self.task_size: self.numclass]
        print(classes)

        trainfolder = self.data_manager.get_dataset(self.train_transform, index=classes, train=True)
        testfolder = self.data_manager.get_dataset(self.test_transform, index=class_set[:self.numclass], train=False)

        self.train_loader = torch.utils.data.DataLoader(trainfolder, batch_size=self.args.batch_size,
            shuffle=True, drop_last=True, num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(testfolder, batch_size=self.args.batch_size,
            shuffle=False, drop_last=False, num_workers=8)

        if current_task > 0:
            self.model.module.Incremental_learning(self.numclass * self.ssl)
        self.model.train()
        if current_task == 0:
            self.model = nn.DataParallel(self.model, device_ids=self.d_ids)

        self.model.to(self.device)

    def train(self, current_task, old_class=0):
        exist_path = self.args.save_path + self.file_name + '/' + str(self.numclass) + '_model.pkl'
        if current_task > 0:
            self.epochs = 61
            self.decay_epochs = 25
            self.learning_rate = self.incre_lr
        if os.path.exists(exist_path):
            # self.model = torch.load(exist_path,map_location={'cuda:1': 'cuda:2'})
            self.model = network_cosine(self.numclass * self.ssl, self.feature_extractor)
            state_dict = torch.load(exist_path)
            self.model.load_state_dict(state_dict)
            self.model = nn.DataParallel(self.model, device_ids=self.d_ids).to(self.device)

            accuracy = 0
            accuracy = self._test(self.test_loader)
            print('accuracy:%.5f' % accuracy)
        else:

            tg_params = self.model.module.get_config_optim(self.learning_rate, self.learning_rate)
            if current_task > 0:
                opt = torch.optim.Adam(tg_params, lr=self.learning_rate, weight_decay=2e-4)
            else:
                custom_weight_decay = 5e-4  # Weight Decay
                custom_momentum = 0.9  # Momentum
                opt = torch.optim.SGD(tg_params, lr=self.learning_rate, momentum=custom_momentum,
                                      weight_decay=custom_weight_decay)
            scheduler = StepLR(opt, step_size=self.decay_epochs, gamma=0.1)
            accuracy = 0
            for epoch in range(self.epochs):
                train_loss = 0
                train_loss1 = 0
                train_loss2 = 0
                train_loss3 = 0
                for step, data in enumerate(self.train_loader):
                    images, target = data
                    images, target = images.to(self.device), target.to(self.device)
                    images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(self.ssl)], 1)
                    images = images.view(-1, 3, self.size, self.size)
                    target = torch.stack([target * self.ssl + k for k in range(self.ssl)], 1).view(-1)
                    opt.zero_grad()
                    loss, loss1, loss2, loss3 = self._compute_loss(images, target, old_class)
                    opt.zero_grad()
                    loss = loss.mean()
                    loss.backward()
                    opt.step()

                    train_loss += loss.item()
                    train_loss1 += loss1.item()
                    train_loss2 += loss2.item()
                    train_loss3 += loss3.item()
                scheduler.step()
                if epoch % self.args.print_freq == 0:
                    accuracy = self._test(self.test_loader)
                    print("lr:", scheduler.get_last_lr())
                    print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))
                    print('Train set: {}, loss_cls: {:.4f}, loss_protoAug: {:.4f},loss_kd: {:.4f},Train Loss: {:.4f}'.format(len(self.train_loader),
                    train_loss1 / (step + 1),train_loss2 / (step + 1),train_loss3 / (step + 1), train_loss / (step + 1)))
        self.protoSave(self.model, self.old_model, self.train_loader, current_task)



    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, data in enumerate(testloader):
            imgs, labels = data
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
                outputs = outputs[:, ::self.ssl]  # only compute predictions on original class nodes
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()
        return accuracy

    def _compute_loss(self, imgs, target, old_class=0):
        output = self.model(imgs)
        output, target = output.to(self.device), target.to(self.device)
        loss_cls = nn.CrossEntropyLoss()(output / self.args.temp, target)
        if self.old_model is None:
            return loss_cls, loss_cls, loss_cls * 0.0, loss_cls * 0.0
        else:
            ###########################################
            feature = self.model.module.feature(imgs)
            feature_old = self.old_model.module.feature(imgs)
            output_old = self.old_model(imgs)
            loss_kd = torch.dist(feature, feature_old, 2)
            ###########################################
            proto_aug = []
            proto_aug_label = []
            index = list(range(old_class))

            proto_aug, proto_aug_label = self.apa(feature, self.prototype, proto_aug, proto_aug_label)

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            soft_feat_aug = self.model.module.fc(proto_aug)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / self.args.temp, proto_aug_label)

            return self.args.cls_weight * loss_cls + self.args.protoAug_weight * loss_protoAug + self.args.kd_weight * loss_kd, loss_cls, loss_protoAug, loss_kd

    def apa(self, feat, proto, proto_aug, proto_aug_label):
        proto_num = proto.shape[0]
        feat_num = feat.shape[0]
        feat = feat.cpu().detach().numpy()
        feat_l2_norm = np.linalg.norm(feat, ord=2, axis=-1)
        proto_l2_norm = np.linalg.norm(proto, ord=2, axis=-1)
        feat_norm = feat / feat_l2_norm.reshape([feat_num, -1])
        proto_norm = proto / proto_l2_norm.reshape([proto_num, -1])
        cos = np.matmul(proto_norm, feat_norm.T)
        max_index = np.argmax(cos, axis=1)
        tag = np.zeros(max_index.shape)
        cos_2 = np.matmul(feat_norm, proto_norm.T)
        max_index_2 = np.argmax(cos_2, axis=1)
        for xx in range(len(max_index)):
            if max_index_2[max_index[xx]] == xx:
                tag[xx] = 1

        feat_select = feat[max_index]
        direct_vec = (feat_select - proto)
        index = list(range(proto_num))

        for j in range(self.args.batch_size * self.rate):
            np.random.shuffle(index)
            random_vec = np.random.normal(0, 1, 512) * self.radius[index[0]]
            d_vec = direct_vec[index[0]]
            denom = np.linalg.norm(d_vec) * np.linalg.norm(random_vec)
            cos_theta = np.dot(d_vec, random_vec) / denom
            if cos_theta > 0.0 and tag[index[0]] != 0:
                p_feature = proto[index[0]] + (1.0 - cos_theta) * random_vec
            else:
                p_feature = proto[index[0]] + random_vec
            proto_aug.append(p_feature)
            proto_aug_label.append(self.ssl * self.class_label[index[0]])

        return proto_aug, proto_aug_label

    def afterTrain(self):
        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        self.numclass += self.task_size
        filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
        torch.save(self.model.module.state_dict(), filename)

        try:
            self.old_model = copy.deepcopy(self.model)
        except:
            torch.save(self.model.module.state_dict(), filename)

        self.old_model.eval()
        current_proto = self.prototype
        return current_proto

    def protoSave(self, model, old_model, loader, current_task):
        features = []
        old_features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                images, targets = data
                feature = model.module.feature(images.to(self.device))
                if self.old_model is not None:
                    old_feature = old_model.module.feature(images.to(self.device))
                    if old_feature.shape[0] == self.args.batch_size:
                        old_features.append(old_feature.cpu().numpy())

                if feature.shape[0] == self.args.batch_size:
                    labels.append(targets.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        if self.old_model is not None:
            print('****compute network drift****')
            old_features = np.array(old_features)
            old_features = np.reshape(old_features,
                                      (old_features.shape[0] * old_features.shape[1], old_features.shape[2]))
            MU = self.prototype
            drift_values = self.tpa(old_features, features, MU)

            MU += drift_values
            self.prototype = MU

        if self.args.density == 1:
            print('density-based prototype generation')
            prototype, radius, class_label, cov_list = self.dbr(features, labels, labels_set)
        else:
            prototype, radius, class_label, cov_list = self.cmp(features, labels, labels_set)

        if current_task == 0:
            self.radius = radius
            self.prototype = np.asarray(prototype)
            self.class_label = class_label
            self.cov_list = cov_list
        else:
            self.prototype = np.concatenate((self.prototype, prototype), axis=0)
            self.class_label = np.concatenate((self.class_label, class_label), axis=0)
            self.radius = np.concatenate((self.radius, radius), axis=0)
            self.cov_list = np.concatenate((self.cov_list, cov_list), axis=0)

        print('radius:', self.radius)
        print('class:', self.class_label)
        print('length of conv list:', len(self.cov_list))
        print('conv shape:', self.cov_list[0].shape)
        print('proto shape:', self.prototype.shape)

    def cmp(self, features, labels, labels_set):
        feature_dim = features.shape[1]
        prototype = []
        radius = []
        class_label = []
        cov_list = []

        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))

            cov = np.cov(feature_classwise.T)
            radius.append(np.sqrt(np.trace(cov) / feature_dim))
            cov_list.append(cov)
        return prototype, radius, class_label, cov_list

    def dbr(self, features, labels, labels_set):
        feature_dim = features.shape[1]

        prototype = []
        radius = []
        class_label = []
        cov_list = []

        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            feature_num = feature_classwise.shape[0]
            ##################################
            l2_norm = np.linalg.norm(feature_classwise, ord=2, axis=-1)
            feature_classwise_norm = feature_classwise / l2_norm.reshape([feature_num, -1])
            cos = np.matmul(feature_classwise_norm, feature_classwise_norm.T)
            euc = 2 - 2 * cos
            extra = np.diag(np.diag(euc))

            euc = np.sqrt(euc - extra + 1e-6)

            avg_dist_k = np.mean(euc, axis=-1)

            feat_w = np.exp(avg_dist_k * self.args.gamma)
            proto = np.sum(feature_classwise * feat_w.reshape([feature_num, -1]), axis=0) / np.sum(feat_w, axis=0)
            prototype.append(proto)

            z = feature_classwise.T - proto.reshape([1, -1]).T  # x - mean
            var = np.sum(z * z * feat_w.reshape([-1, feature_num]), axis=-1) / (np.sum(feat_w, axis=0) - 1)

            cov = np.cov(feature_classwise.T)
            radius.append(np.sqrt(np.mean(var)))
            cov_list.append(cov)
        return prototype, radius, class_label, cov_list

    def get_k_max(self, array, k):
        _k_sort = np.argpartition(array, -k, axis=-1)[:, -k:]  # 最大的k个数据的下标
        new_array = array[np.arange(array.shape[0])[:, None], _k_sort]
        array_min = np.min(new_array, axis=-1)
        array_min = array_min.reshape((-1, 1))
        array_min = np.tile(array_min, [1, array.shape[-1]])
        test_array = array - array_min
        k_array = np.where(test_array < 0.0, 0.0, array)
        return k_array

    def tpa(self, old_feat, feat, proto_old):
        DY = feat - old_feat
        # ()
        proto_old = np.array(proto_old)
        proto_num = proto_old.shape[0]
        feat_num = old_feat.shape[0]
        feat_l2_norm = np.linalg.norm(old_feat, ord=2, axis=-1)
        proto_l2_norm = np.linalg.norm(proto_old, ord=2, axis=-1)
        old_feat_norm = old_feat / feat_l2_norm.reshape([feat_num, -1])
        proto_old_norm = proto_old / proto_l2_norm.reshape([proto_num, -1])

        distance = np.sum(
            (np.tile(old_feat_norm[None, :, :], [proto_old_norm.shape[0], 1, 1]) - np.tile(proto_old_norm[:, None, :],
                                                                                           [1, old_feat_norm.shape[0],
                                                                                            1])) ** 2, axis=2)

        W = np.exp(-distance * self.args.alpha_2)

        W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        displacement = np.sum(
            np.tile(W_norm[:, :, None], [1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        cos = np.matmul(proto_old_norm, proto_old_norm.T)
        euc = 2 - 2 * cos
        extra = np.diag(np.diag(euc))
        dis_in_proto = np.sqrt(euc - extra)
        W_p = np.exp(-dis_in_proto * self.args.alpha)
        W_p_extra = np.diag(np.diag(W_p) * 1)
        W_p = W_p - W_p_extra
        W_p_k = self.get_k_max(W_p, k=10)
        W_p_norm = W_p_k / np.tile(np.sum(W_p_k, axis=1)[:, None], [1, W_p_k.shape[1]])
        displacement_p = np.sum(
            np.tile(W_p_norm[:, :, None], [1, 1, displacement.shape[1]]) * np.tile(displacement[None, :, :],
                                                                                   [W_p.shape[0], 1, 1]), axis=1)
        return (displacement + displacement_p * self.args.beta) * self.drift_weight