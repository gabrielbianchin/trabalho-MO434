import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.CamVid import CamVid
import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
from loss import DiceLoss
from data_loader import SegDataset


def val(args, model, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = predict.cpu().numpy()

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = label.cpu().numpy()

            # compute per pixel accuracy

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist)[:-1]
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)
        return precision, miou


def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss()
    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            output, output_sup1, output_sup2 = model(data)
            loss1 = loss_func(output, label.squeeze())
            loss2 = loss_func(output_sup1, label.squeeze())
            loss3 = loss_func(output_sup2, label.squeeze())
            loss = loss1 + loss2 + loss3
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
#             if i % 100 == 0:
#                 viz.image(data[0], win=0)
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_model_path, 'latest_model.pth'))

        if epoch % args.validation_step == 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'best_model.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)


def main(params, viz):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')

    args = parser.parse_args(params)

    # Define the augmentation pipeline
    aug_pipeline = A.Compose([
    #     A.OneOf([
    # #         A.Blur(p=0.5),
    #         A.RandomBrightnessContrast(p=0.5),    
    # #         A.RandomGamma(p=0.5),
    #         A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, p=0.5)
    #     ], p=0.5),
        A.HorizontalFlip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
    ])

    X_train = pd.read_pickle('/home/gustavu92/Desktop/unicamp/data/bonn/X_train.pkl')
    X_val = pd.read_pickle('/home/gustavu92/Desktop/unicamp/data/bonn/X_val.pkl')
    y_train = pd.read_pickle('/home/gustavu92/Desktop/unicamp/data/bonn/y_train.pkl')
    y_val = pd.read_pickle('/home/gustavu92/Desktop/unicamp/data/bonn/y_val.pkl')

    shape = (args.crop_width, args.crop_height)

    train_generator = SegDataset(X_train, y_train, shape, None)
    val_generator = SegDataset(X_val, y_val, shape, None)


    dataloader_train = DataLoader(
        train_generator,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    dataloader_val = DataLoader(
        val_generator,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val)

    # val(args, model, dataloader_val, csv_path)


if __name__ == '__main__':
    DEFAULT_PORT = 8097
    DEFAULT_HOSTNAME = "http://localhost"
    parser = argparse.ArgumentParser(description='Demo arguments')
    parser.add_argument('-port', metavar='port', type=int, default=DEFAULT_PORT,
                        help='port the visdom server is running on.')
    parser.add_argument('-server', metavar='server', type=str,
                        default=DEFAULT_HOSTNAME,
                        help='Server address of the target to run the demo on.')
    parser.add_argument('-base_url', metavar='base_url', type=str,
                    default='/',
                    help='Base Url.')
    parser.add_argument('-username', metavar='username', type=str,
                    default='',
                    help='username.')
    parser.add_argument('-password', metavar='password', type=str,
                    default='',
                    help='password.')
    parser.add_argument('-use_incoming_socket', metavar='use_incoming_socket', type=bool,
                    default=True,
                    help='use_incoming_socket.')
    FLAGS = parser.parse_args()
    try:
        viz = Visdom(port=FLAGS.port, server=FLAGS.server, base_url=FLAGS.base_url, username=FLAGS.username, password=FLAGS.password, use_incoming_socket=FLAGS.use_incoming_socket)
    except Exception as e:
        print("Visdom not initialized")
        
    params = [
        '--num_epochs', '1000',
        '--learning_rate', '2.5e-3',
        '--num_workers', '6',
        '--num_classes', '3',
        '--cuda', '0',
        '--batch_size', '4',  # 6 for resnet101, 12 for resnet18
        '--save_model_path', './checkpoints',
        '--context_path', 'resnet101',  # only support resnet18 and resnet101
        '--optimizer', 'adam',
        '--loss', 'crossentropy',
        '--crop_width', '512',
        '--crop_height', '384',
    ]
    main(params, viz)

