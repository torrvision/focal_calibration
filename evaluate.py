import os
import sys
import torch
import random
import argparse
from torch import nn
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

# Import dataloaders
import Data.cifar10 as cifar10
import Data.cifar100 as cifar100
import Data.tiny_imagenet as tiny_imagenet

# Import network architectures
from Net.resnet_tiny_imagenet import resnet50 as resnet50_ti
from Net.resnet import resnet50, resnet110
from Net.wide_resnet import wide_resnet_cifar
from Net.densenet import densenet121

# Import metrics to compute
from Metrics.metrics import test_classification_net_logits
from Metrics.metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss

# Import temperature scaling and NLL utilities
from temperature_scaling import ModelWithTemperature


# Dataset params
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200
}

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'tiny_imagenet': tiny_imagenet
}

# Mapping model name to model function
models = {
    'resnet50': resnet50,
    'resnet50_ti': resnet50_ti,
    'resnet110': resnet110,
    'wide_resnet': wide_resnet_cifar,
    'densenet121': densenet121
}


def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = './'
    model = 'resnet50'
    save_loc = './'
    saved_model_name = 'resnet50_cross_entropy_350.model'
    num_bins = 15
    model_name = None
    train_batch_size = 128
    test_batch_size = 128
    cross_validation_error = 'ece'

    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to test on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name", help='name of the model')
    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to test')
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to import the model')
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--num-bins", type=int, default=num_bins, dest="num_bins",
                        help='Number of bins')
    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("-da", action="store_true", dest="data_aug",
                        help="Using data augmentation")
    parser.set_defaults(data_aug=True)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("--cverror", type=str, default=cross_validation_error,
                        dest="cross_validation_error", help='Error function to do temp scaling')
    parser.add_argument("-log", action="store_true", dest="log",
                        help="whether to print log data")

    return parser.parse_args()


def get_logits_labels(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    return logits, labels


if __name__ == "__main__":

    # Checking if GPU is available
    cuda = False
    if (torch.cuda.is_available()):
        cuda = True

    # Setting additional parameters
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    args = parseArgs()

    if args.model_name is None:
        args.model_name = args.model

    dataset = args.dataset
    dataset_root = args.dataset_root
    model_name = args.model_name
    save_loc = args.save_loc
    saved_model_name = args.saved_model_name
    num_bins = args.num_bins
    cross_validation_error = args.cross_validation_error

    # Taking input for the dataset
    num_classes = dataset_num_classes[dataset]
    if (args.dataset == 'tiny_imagenet'):
        val_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu)

        test_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu)
    else:
        _, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=1,
            pin_memory=args.gpu
        )

        test_loader = dataset_loader[args.dataset].get_test_loader(
            batch_size=args.test_batch_size,
            pin_memory=args.gpu
        )

    model = models[model_name]

    net = model(num_classes=num_classes, temp=1.0)
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    net.load_state_dict(torch.load(args.save_loc + args.saved_model_name))

    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()
    adaece_criterion = AdaptiveECELoss().cuda()
    cece_criterion = ClasswiseECELoss().cuda()

    logits, labels = get_logits_labels(test_loader, net)
    conf_matrix, p_accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    p_ece = ece_criterion(logits, labels).item()
    p_adaece = adaece_criterion(logits, labels).item()
    p_cece = cece_criterion(logits, labels).item()
    p_nll = nll_criterion(logits, labels).item()

    res_str = '{:s}&{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}'.format(saved_model_name,  1-p_accuracy,  p_nll,  p_ece,  p_adaece, p_cece)

    # Printing the required evaluation metrics
    if args.log:
        print (conf_matrix)
        print ('Test error: ' + str((1 - p_accuracy)))
        print ('Test NLL: ' + str(p_nll))
        print ('ECE: ' + str(p_ece))
        print ('AdaECE: ' + str(p_adaece))
        print ('Classwise ECE: ' + str(p_cece))


    scaled_model = ModelWithTemperature(net, args.log)
    scaled_model.set_temperature(val_loader, cross_validate=cross_validation_error)
    T_opt = scaled_model.get_temperature()
    logits, labels = get_logits_labels(test_loader, scaled_model)
    conf_matrix, accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    ece = ece_criterion(logits, labels).item()
    adaece = adaece_criterion(logits, labels).item()
    cece = cece_criterion(logits, labels).item()
    nll = nll_criterion(logits, labels).item()

    res_str += '&{:.4f}({:.2f})&{:.4f}&{:.4f}&{:.4f}'.format(nll,  T_opt,  ece,  adaece, cece)

    if args.log:
        print ('Optimal temperature: ' + str(T_opt))
        print (conf_matrix)
        print ('Test error: ' + str((1 - accuracy)))
        print ('Test NLL: ' + str(nll))
        print ('ECE: ' + str(ece))
        print ('AdaECE: ' + str(adaece))
        print ('Classwise ECE: ' + str(cece))

    # Test NLL & ECE & AdaECE & Classwise ECE
    print(res_str)