import argparse
import os
import numpy as np
from torchvision import transforms
import time


from tqdm import tqdm
from utils.helpers import gen_train_dirs, plot_confusion_matrix, get_train_trans, get_test_trans
from utils.routines import train_epoch, evaluate
from datasets.cityscapes_ext import CityscapesExt


from torch.utils.data import DataLoader, ConcatDataset

from datasets.nighttime_driving import NighttimeDrivingDataset
from datasets.dark_zurich import DarkZurichDataset
from models.refinenet import RefineNet
from datasets.foggy_driving import foggyDrivingDataset
from datasets.foggy_driving_full import foggyDrivingFullDataset
from datasets.foggy_zurich import foggyZurichDataset
from datasets.overcast import overcastDataset
from datasets.acdc_fog import ACDCFogDataset
from datasets.acdc_night import ACDCNightDataset
from datasets.acdc_night_cont import ACDCNightContDataset
from datasets.acdc_rain import ACDCRainDataset
from datasets.acdc_snow import ACDCSnowDataset
from mypath import Path
from dataloaders import make_data_loader
from dataloaders.custom_transforms import denormalizeimage
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from datasets.cityscapes_ext import CityscapesExt
import network
import matplotlib.pyplot as plt
from datasets.foggy_driving import foggyDrivingDataset
from datasets.foggy_driving_full import foggyDrivingFullDataset
from datasets.nighttime_driving import NighttimeDrivingDataset
from datasets.dark_zurich import DarkZurichDataset
from DenseNCLoss_3d import DenseNCLoss
from torch.autograd import Variable
from datasets.C_driving_cloudy import CDrivingCloudDataset
from datasets.C_driving_rainy import CDrivingRainDataset
from datasets.C_driving_snowy import CDrivingSnowDataset
from datasets.C_driving_overcast import CDrivingOvercastDataset
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
torch.cuda.empty_cache()
import shutil, time, random
import matplotlib.pyplot as plt
import numpy as np

miou_nd=[]
iteration_list=[]

#@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    loss= -(x.cpu().softmax(1) * x.cpu().log_softmax(1)).sum(1).mean(0)
    #print(loss.size())
    loss=torch.sum(loss)
    return Variable(torch.tensor([loss]), requires_grad=True)


class MiouTracker:
    def __init__ (self, csv_file_name):
        import csv
        self.mious = {}
        self.file = open(csv_file_name, 'w')
        self.writer = csv.writer(self.file, delimiter='\t')
        self.writer.writerow(['i', 'day', 'fog', 'night', 'rain', 'snow', 'avg', 'w_avg'])
        self.file.flush()
        print ('ok')
        self.cur_row = [0 for i in range (8)]


    def update (self, counter, epoch, trainer, dataloaders):
        self.cur_row[0] = counter // 400
        
        if (counter%100 == 0):
            if (counter == 0):
                self.test_fog (epoch, trainer, dataloaders)
                self.test_night (epoch, trainer, dataloaders)
                self.test_rain (epoch, trainer, dataloaders)
                self.test_snow (epoch, trainer, dataloaders)
                self.test_city_val (epoch, trainer, dataloaders)
                self.write_row ()
            else:
                if ((counter % 400) == 0):
                    self.test_snow (epoch, trainer, dataloaders)
                    self.test_city_val (epoch, trainer, dataloaders)
                    self.write_row ()
                elif ((counter % 400) == 100): self.test_fog (epoch, trainer, dataloaders)
                elif ((counter % 400) == 200): self.test_night (epoch, trainer, dataloaders)
                elif ((counter % 400) == 300): self.test_rain (epoch, trainer, dataloaders)


    
    def write_row (self):
        self.cur_row[6] = sum (self.cur_row[1:6]) / 5
        self.cur_row[7] = sum (self.cur_row[2:6]) / 4
        for i in range (1, len (self.cur_row)):
            self.cur_row[i] = round (100*self.cur_row[i], 2)
        print(self.cur_row)
        self.writer.writerow (self.cur_row)
        self.file.flush()
    
    def test_fog (self, epoch, trainer, dataloaders):
        self.cur_row[2] = trainer.validation_nd(epoch,dataloaders['testset_acdc_fog'])
        #print("fog")
        #print(self.cur_row[2])  
    def test_snow (self, epoch, trainer, dataloaders):
        self.cur_row[5] = trainer.validation_nd(epoch,dataloaders['testset_acdc_snow'])
        #print("snow")
        #print(self.cur_row[5])  
    def test_city_val (self, epoch, trainer, dataloaders):
        self.cur_row[1] = trainer.validation_nd(epoch,dataloaders['val'])
        #print("day")
        #print(self.cur_row[1])  
    def test_rain (self, epoch, trainer, dataloaders):
        self.cur_row[4] = trainer.validation_nd(epoch,dataloaders['testset_acdc_rain'])
        #print("rain")
        #print(self.cur_row[4])  
    def test_night (self, epoch, trainer, dataloaders):
        self.cur_row[3] = trainer.validation_nd(epoch,dataloaders['testset_acdc_night'])
        #print("night")
        #print(self.cur_row[3])  
    def close (self):
        self.file.close()
                    
        






        

        
    
    




class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
        }

        # Define network
        if args.base_model=='deeplabv3+_mobilenet':
            model = network.deeplabv3plus_mobilenet(num_classes=self.nclass, output_stride=args.out_stride)
        if args.base_model=='deeplabv3+_resnet101':
            model = network.deeplabv3plus_resnet101(num_classes=self.nclass, output_stride=args.out_stride)
        if args.base_model=='deeplabv3plus_resnet50':
            model = network.deeplabv3plus_resnet50(num_classes=self.nclass, output_stride=args.out_stride)
        if args.base_model=='deeplabv3_resnet50':
            model = network.deeplabv3_resnet50(num_classes=self.nclass, output_stride=args.out_stride)
        #print(model)
        #checkpoint = torch.load('/home/nikhil/scratch/CIConv_zero_shot/experiments/3_segmentation/runs/20211014-212540/weights/checkpoint.pth')

        #if torch.cuda.is_available():
        #    model = torch.nn.DataParallel(model).cuda()
        #    print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
        #model.load_state_dict(checkpoint['model_state'], strict=True)
        #train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
        #                {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        #optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
            #                        weight_decay=args.weight_decay, nesterov=args.nesterov)
        #print(model.backbone)
        
        optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr},
        ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model.cuda(), optimizer
        
        if args.densencloss >0:
            self.densenclosslayer = DenseNCLoss(weight=args.densencloss, sigma_rgb=args.sigma_rgb, sigma_xy=args.sigma_xy, scale_factor=args.rloss_scale)
            print(self.densenclosslayer)
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        # self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, 1, 500)

        # Using cuda
        #if args.cuda:
        #    self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
        #    patch_replication_callback(self.model)
        #    self.model = self.model.cuda()
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = 108
            #args.start_epoch = checkpoint['epoch']
            if args.cuda and  args.base_model!='deeplabv3plus_resnet50':
                self.model.load_state_dict(checkpoint['model_state'])
                #self.model.load_state_dict(checkpoint['state_dict'])
                #print(checkpoint['state_dict'])
                #self.model.module.load_state_dict(checkpoint['state_dict'],strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            #if not args.ft:
            ##    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
             #   self.best_pred = checkpoint['best_score']
            #print("=> loaded checkpoint '{}' (epoch {})"
            #      .format(args.resume, checkpoint['cur_itrs']))


        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def test_time_training(self,trainer,dataloaders, epoch,loader):
        import buffer
        buffer_size = 20
        # replay_period = 15
        replay_period = buffer_size//2
        # update_period = 5
        update_period = buffer_size//4

        softmax = nn.Softmax(dim=1)

        def priority_fn (image, label, output):
            '''
            Takes a [3, H, W] shaped single image (pytorch tensor),
            corresponding [H, W] label pytorch tensor, and [19, H, W] 
            shaped model's output tensor to calculate the priority of 
            a DataPoint. Higher the floating point number returned by 
            this function, higher the priority of that datapoint in the buffer.
            
            Return: float: priority of a DataPoint
            '''
            images = torch.unsqueeze (image, 0)
            output = torch.unsqueeze (output, 0)
            denormalized_image = denormalizeimage(images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            croppings = torch.ones ((images.shape[0], images.shape[2], images.shape[3]), dtype=torch.float32)
            probs = softmax(output)

            with torch.no_grad():
                densencloss = self.densenclosslayer(denormalized_image,probs,croppings)
            # if self.args.cuda:
            #     densencloss = densencloss.cuda()
            #loss = 0.000001*celoss.cuda() + densencloss
            # loss = -1e-1*densencloss
            # print(-1e-1 * float(densencloss.item()))
            return -float(densencloss.item())

            # return softmax_entropy (output)
        
        if (self.args.buffer=='random'):
            buffer = buffer.BufferManager (buffer_size, replay_period, update_period,num_update=1, buffer='random')
        elif (self.args.buffer=='priority'):
            buffer = buffer.BufferManager (buffer_size, replay_period, update_period,num_update=update_period, buffer='priority')
        elif (self.args.buffer=='priority_nc'):
            buffer = buffer.BufferManager (buffer_size, replay_period, update_period,num_update=update_period, buffer='priority',priority_fn=priority_fn)
        else:
            raise NotImplementedError

        counter = 0
        criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)
        #for i, sample in enumerate(tbar):
        #for i, (inputs, labels) in enumerate(loader):
        weights = -1*torch.log(torch.FloatTensor(
        [0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341, 0.00207795, 0.0055127, 0.15928651,
         0.01157818, 0.04018982, 0.01218957, 0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
         0.00413907])).cuda()
        weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.16 + 1.0
        weights = weights.cuda()
        interp = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)
        train_loss = 0.0
        train_celoss = 0.0
        train_ncloss = 0.0
        iteration=(epoch-108)*5
        #num_img_tr = len(self.train_loader)
        num_img_tr = len(loader)
        mt = MiouTracker(self.args.csvlog)
        mt.update (0, epoch, trainer, dataloaders)  # initial miou scores
        # sys.exit()

        for iteration in range (self.args.num_iter):
            tbar = tqdm(loader)
            for i, (inputs, labels,filepath) in enumerate(tbar):
                image = inputs.float()
                target = labels.long()
                counter += image.shape[0]
                iteration = iteration+1
                iteration_list.append(iteration)
                self.model.train()
                with torch.no_grad():
                    output = self.model (image.cuda()).cpu()
                # print ('reach counter = {}'.format (counter))
                # output = output.cpu()
                image = image.cpu()
                target = target.cpu()
                replay_dataset = buffer.update (image, target, output)
                if(replay_dataset is None):
                    continue
                buffer_dl = torch.utils.data.DataLoader(replay_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=False, num_workers=self.args.workers)
                for i, image_label_pairs in enumerate (buffer_dl):
                    images, target = image_label_pairs
                    images = images.cuda()
                    # target = target.cuda()
                    # Pixels labeled 255 are those unlabeled pixels. Padded region are labeled 254.
                    # see function RandomScaleCrop in dataloaders/custom_transforms.py for the detail in data preprocessing
                    #croppings = (target!=254).float()
                    # target[target==254]=255
                    croppings = torch.ones ((images.shape[0], images.shape[2], images.shape[3]), dtype=torch.float32)
                    # self.scheduler(self.optimizer, i, 0, self.best_pred)
                    self.optimizer.zero_grad()
                    #self.model.eval()
                    self.model.eval()
                    output = self.model(images)
                    #weights_prob = weights.expand(output.size()[0], output.size()[3], output.size()[2], 19)
                    #weights_prob = weights_prob.transpose(1, 3)
                    #output3 = interp(output * weights_prob)
                    # print ('i = {} input.shape = {} output.shape = {} target.shape = {}'.format (i, images.shape, output.shape, target.shape))
                    # sys.exit()
                    self.model.train()
                    # celoss = softmax_entropy(output).cuda()
                    if self.args.densencloss ==0:
                        loss = celoss
                    else:
                        #output for normal predictions
                        probs = softmax(output)
                        #print(croppings.shape)
                        #exit()
                        #miou=trainer.validation_nd(epoch,dataloaders['test_nd'])
                        #miou_nd.append(miou*100)
                        start_time = time.perf_counter ()
                        denormalized_image = denormalizeimage(images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        # print ('len(denormalized_image) = {}'.format (denormalized_image.shape[0]))
                        # print ('len(probs) = {}'.format (probs.shape[0]))
                        densencloss = self.densenclosslayer(denormalized_image,probs,croppings)
                        print(densencloss)
                        if self.args.cuda:
                            densencloss = densencloss.cuda()
                        #loss = 0.000001*celoss.cuda() + densencloss
                        loss = -1e-9*densencloss
                        train_ncloss += densencloss.item()

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    end_time = time.perf_counter ()
                    #print(end_time - start_time, "seconds")
                    self.optimizer.zero_grad()
                    train_loss += loss.item()

                #trainer.validation(epoch)
                mt.update(counter, epoch, trainer, dataloaders)

                tbar.set_description('Train loss: %.3f = CE loss %.3f + NC loss: %.3f' 
                                    % (train_loss / (i + 1),train_celoss / (i + 1),train_ncloss / (i + 1)))
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        mt.close()
        # self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        # print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        # print('Loss: %.3f' % train_loss)

        if self.args.save_interval:
            # save checkpoint every interval epoch
            is_best = False
            if (epoch + 1) % self.args.save_interval == 0:
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best, filename='checkpoint_epoch_{}.pth.tar'.format(str(epoch+1)))

        sys.exit()

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            target[target==254]=255
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation_nd(self,epoch,data):
        mean = (0.485, 0.456, 0.406) 
        std = (0.229, 0.224, 0.225) 
        target_size = (512,1024)
        crop_size = (384,768)
        #print(CityscapesExt.voidClass)
        test_acc_nd, test_loss_nd, miou_nd, confmat_nd, iousum_nd = evaluate(data,
            self.model, self.criterion, epoch, CityscapesExt.classLabels, CityscapesExt.validClasses,void=CityscapesExt.voidClass,optimizer=self.optimizer,
             maskColors=CityscapesExt.maskColors, mean=mean, std=std)
        return miou_nd

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--test', type=int, default=1,required=True,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    parser.add_argument('--base_model', type=str, default='deeplabv3+_mobilenet',
                        help='Base model')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # model saving option
    parser.add_argument('--save-interval', type=int, default=None,
                        help='save model interval in epochs')


    # rloss options
    parser.add_argument('--densencloss', type=float, default=0,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--rloss-scale',type=float,default=1.0,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb',type=float,default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy',type=float,default=80.0,
                        help='DenseCRF sigma_xy')

    parser.add_argument('--buffer', type=str, choices=['random', 'priority', 'priority_nc'],
                        help='Replay buffer type for continual learning')
    parser.add_argument('--csvlog', type=str, default="test.log",
                        help='Continual learning miou scores will be saved in this csv log file')
    parser.add_argument('--num_iter', type=int, default=10,
                        help='Number of iterations for continual learning setup (default: 10)')

    transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()])
    cs_path = '/home/nikhil/scratch/datasets/cityscapes/'
    nd_path = '/home/nikhil/scratch/datasets/NighttimeDrivingTest/'
    dz_path = '/home/nikhil/scratch/datasets/Dark_Zurich_val_anon/'
    fd_path= '/home/nikhil/scratch/datasets/Foggy_Driving/'
    fz_path = '/home/nikhil/scratch/datasets/Foggy_Driving_Full/'
    fz_actual_path = '/home/nikhil/scratch/datasets/Foggy_Zurich/'
    oc_actual_path = '/home/nikhil/scratch/datasets/'
    acdc_path='/home/nikhil/scratch/datasets/acdc_dataset/'
    C_driving_path = '/home/nikhil/scratch/abhishek/C-Driving'
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225) 
    target_size = (512,1024)
    crop_size = (384,768)
    args = parser.parse_args()
    train_trans = get_train_trans(mean, std, target_size)
    test_trans = get_test_trans(mean, std, target_size)
    trainset = CityscapesExt(cs_path, split='train', target_type='semantic', transforms=train_trans)
    #trainset = CityscapesExt(cs_path, split='train', target_type='semantic')
    valset = CityscapesExt(cs_path, split='val', target_type='semantic', transforms=test_trans)
    testset_day = CityscapesExt(cs_path, split='test', target_type='semantic', transforms=test_trans)
    testset_nd = NighttimeDrivingDataset(nd_path, transforms=test_trans)
    testset_dz = DarkZurichDataset(dz_path, transforms=test_trans)
    testset_fd = foggyDrivingDataset(fd_path, transforms=test_trans)
    testset_fz = foggyDrivingFullDataset(fz_path, transforms=test_trans)
    testset_fz_actual = foggyZurichDataset(fz_actual_path, transforms=test_trans)
    test_sets = torch.utils.data.ConcatDataset([testset_nd, testset_dz,testset_fd,testset_fz,testset_fz_actual])
    train_dev_loader = DataLoader(dataset=test_sets,batch_size=args.batch_size,shuffle=True)
    testset_dz_full = DarkZurichDataset(dz_path,split='test', transforms=test_trans)
    testset_oc = overcastDataset(oc_actual_path, transforms=test_trans)
    testset_acdc_fog = ACDCFogDataset(acdc_path,split='val', transforms=test_trans)
    testset_acdc_rain = ACDCRainDataset(acdc_path,split='val', transforms=test_trans)
    testset_acdc_snow = ACDCSnowDataset(acdc_path,split='val', transforms=test_trans)
    testset_acdc_night_cont = ACDCNightContDataset(acdc_path,split='val', transforms=test_trans)
    testset_acdc_night = ACDCNightDataset(acdc_path,split='val', transforms=test_trans)
    test_sets_acdc = torch.utils.data.ConcatDataset([testset_acdc_night, testset_acdc_snow,testset_acdc_rain,testset_acdc_fog])
    train_dev_loader_acdc = DataLoader(dataset=test_sets,batch_size=args.batch_size,shuffle=True)
    valset_c_driving_cloudy = CDrivingCloudDataset(C_driving_path,split='val',transforms = test_trans)
    valset_c_driving_rainy = CDrivingRainDataset(C_driving_path,split='val',transforms = test_trans)
    valset_c_driving_snowy = CDrivingSnowDataset(C_driving_path,split='val',transforms = test_trans)
    valset_c_driving_overcast = CDrivingOvercastDataset(C_driving_path,split='val',transforms = test_trans)
    valset_c_driving = torch.utils.data.ConcatDataset([valset_c_driving_cloudy, valset_c_driving_rainy, valset_c_driving_snowy, valset_c_driving_overcast])
    
    #new order
    # continual_train = torch.utils.data.ConcatDataset([testset_acdc_fog, testset_acdc_snow, valset, testset_acdc_rain, testset_acdc_night])
    
    #old order
    # continual_train = torch.utils.data.ConcatDataset([valset, testset_acdc_fog, testset_acdc_snow, testset_acdc_rain, testset_acdc_night])
    continual_train = torch.utils.data.ConcatDataset([testset_acdc_fog, testset_acdc_night_cont, testset_acdc_rain, testset_acdc_snow])  #CoTTA order
    continual_test = torch.utils.data.ConcatDataset([testset_acdc_fog, testset_acdc_night, testset_acdc_rain, testset_acdc_snow])  #CoTTA order
    # continual_test = continual_train

    #print("done")
    #exit()
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    dataloaders['val'] = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_day'] = torch.utils.data.DataLoader(testset_day, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_nd'] = torch.utils.data.DataLoader(testset_nd, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_dz'] = torch.utils.data.DataLoader(testset_dz, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_fdd'] = torch.utils.data.DataLoader(testset_fd, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_fd'] = torch.utils.data.DataLoader(testset_fz, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_dz_full'] = torch.utils.data.DataLoader(testset_dz_full, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_fz'] = torch.utils.data.DataLoader(testset_fz_actual, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    #dataloaders['testset_oc'] = torch.utils.data.DataLoader(testset_oc, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    #print(len(dataloaders))
    dataloaders['testset_acdc_fog'] = torch.utils.data.DataLoader(testset_acdc_fog, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['testset_acdc_rain'] = torch.utils.data.DataLoader(testset_acdc_rain, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['testset_acdc_snow'] = torch.utils.data.DataLoader(testset_acdc_snow, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['testset_acdc_night'] = torch.utils.data.DataLoader(testset_acdc_night, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['valset_c_driving_cloudy'] = torch.utils.data.DataLoader(valset_c_driving_cloudy, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['valset_c_driving_rainy'] = torch.utils.data.DataLoader(valset_c_driving_rainy, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['valset_c_driving_snowy'] = torch.utils.data.DataLoader(valset_c_driving_snowy, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['valset_c_driving_overcast'] = torch.utils.data.DataLoader(valset_c_driving_overcast, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['continual_train'] = torch.utils.data.DataLoader(continual_train, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['continual_test'] = torch.utils.data.DataLoader(continual_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    
    num_classes = len(CityscapesExt.validClasses)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 130,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.001,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    epoch=0

    # print ("my miou = ", trainer.validation_nd(108,dataloaders['val']))
    print('Starting Epoch:', trainer.args.start_epoch)
    #print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        if args.test==0:
             miou=trainer.test_time_training_source(trainer,dataloaders,epoch)
        if args.test==1:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_nd'])
        if args.test==2:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_dz'])
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_nd'])
             #miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=train_dev_loader)
             #miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_dz_full'])
        if args.test==3:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_fdd'])
        if args.test==4:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_fd'])
        if args.test==5:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_fz'])
        if args.test==6:
            print("whole data")
            miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=train_dev_loader)
        if args.test==7:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['testset_acdc_fog'])
        if args.test==8:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['testset_acdc_rain'])
        if args.test==9:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['testset_acdc_snow'])
        if args.test==10:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['testset_acdc_night'])
        #trainer.test_time_training(trainer,dataloaders,epoch)
        if args.test==11:
            print("whole data acdc")
            miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=train_dev_loader_acdc)
        if args.test==12:
            print("C_driving cloudy")
            miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['valset_c_driving_cloudy'])
        if args.test==13:
            print("C_driving rainy")
            miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['valset_c_driving_rainy'])
        if args.test==14:
            print("C_driving snowy")
            miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['valset_c_driving_snowy'])
        if args.test==15:
            print("C_driving overcast")
            miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['valset_c_driving_overcast'])
        if args.test==16:
            print ("\n\nContinual\n")
            print('number of images in train set =', len(dataloaders['continual_train'].dataset))
            print('number of images in test set =', len(dataloaders['continual_test'].dataset))
            miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['continual_train'])

        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
            #trainer.validation_nd(epoch,dataloaders['test_nd'])
            #trainer.validation_nd(epoch,dataloaders['test_dz'])
            #print('Foggy Driving Dense mIOU')
            #miou=trainer.test_time_training(epoch,loader=dataloaders['test_fd'])
            '''
            print("epoch")
            print('Night Driving mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_nd'])
            print(miou)
            print('Dark Zurich mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_dz'])
            print(miou)
            print('Foggy Driving Dense mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fdd'])
            print(miou)
            print('Foggy Driving Full mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fd'])
            print(miou)
            print('Foggy Zurich mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fz'])
            print(miou)
            '''
    #plt.xlabel('Number of Iterations')
        # naming the y axis
    #plt.ylabel('mIOU Night Driving')
        # giving a title to my graph
    print(iteration_list)
    print(miou_nd)
    plt.title('Night Driving')
    plt.plot(iteration_list, miou_nd)
    #function to show the plot
    plt.savefig('miou_nd'+str(epoch)+'.png')
    trainer.writer.close()




if __name__ == "__main__":
   main()
