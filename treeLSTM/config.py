import argparse

def parse_args(type=0):
    if type == 0:
        parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
        parser.add_argument('--data', default='data/sick/',
                            help='path to dataset')
        parser.add_argument('--glove', default='data/glove/',
                            help='directory with GLOVE embeddings')
        parser.add_argument('--batchsize', default=25, type=int,
                            help='batchsize for optimizer updates')
        parser.add_argument('--epochs', default=15, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=0.01, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--wd', default=1e-4, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--optim', default='adam',
                            help='optimizer (default: adam)')
        parser.add_argument('--seed', default=123, type=int,
                            help='random seed (default: 123)')
        cuda_parser = parser.add_mutually_exclusive_group(required=False)
        cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
        cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
        parser.set_defaults(cuda=True)

        args = parser.parse_args()
        return args
    else: # for sentiment classification on SST
        parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment Analysis Trees')
        parser.add_argument('--name', default='default_name',
                            help='name for log and saved models')
        parser.add_argument('--saved', default='./',
                            help='name for log and saved models')

        parser.add_argument('--model_name', default='constituency',
                            help='model name constituency or dependency')
        parser.add_argument('--data', default='treeLSTM/data/sst/',
                            help='path to dataset')
        parser.add_argument('--glove', default='treeLSTM/data/glove/',
                            help='directory with GLOVE embeddings')
        parser.add_argument('--batchsize', default=25, type=int,
                            help='batchsize for optimizer updates')
        parser.add_argument('--epochs', default=10, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=0.05, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--emblr', default=0.1, type=float,
                            metavar='EMLR', help='initial embedding learning rate')
        parser.add_argument('--wd', default=1e-4, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--reg', default=1e-4, type=float,
                            help='l2 regularization (default: 1e-4)')
        parser.add_argument('--optim', default='adagrad',
                            help='optimizer (default: adagrad)')
        parser.add_argument('--seed', default=123, type=int,
                            help='random seed (default: 123)')
        parser.add_argument('--fine_grain', default=0, type=int,
                            help='fine grained (default 0 - binary mode)')
        parser.add_argument("--loss", type=str, default='cross_entropy', dest="loss_function",
                    help="Loss function to be used for training")
        parser.add_argument("--gamma", type=float, default=3.0,
                            dest="gamma", help="Gamma for focal components")
        parser.add_argument("--gamma-schedule", type=int, default=0,
                            dest="gamma_schedule", help="Schedule gamma or not")
        parser.add_argument("--gamma2", type=float, default=1.0,
                            dest="gamma2", help="Gamma for different focal components")
        parser.add_argument("--gamma3", type=float, default=1.0,
                            dest="gamma3", help="Gamma for different focal components")
        parser.add_argument("--lamda", type=float, default=8.0,
                            dest="lamda", help="Regularization factor")
        parser.add_argument("--smoothing", type=float, default=0.0,
                            dest="smoothing", help="Smoothing factor for labels")
        parser.add_argument("--num-bins", type=int, default=15, dest="num_bins",  
                            help='Number of bins')
        parser.add_argument("-cn", action="store_true", dest="cross_validate_on_nll",
                            help="cross validate on nll")
        parser.set_defaults(cross_validate_on_nll=False)
        parser.add_argument("-ce", action="store_true", dest="cross_validate_on_ece",
                            help="cross validate on ece")
        parser.set_defaults(cross_validate_on_ece=False)
        parser.add_argument("-cae", action="store_true", dest="cross_validate_on_adaece",
                            help="cross validate on adaptive ece")
        parser.set_defaults(cross_validate_on_adaece=False)
        parser.add_argument("-se", action="store_true", dest="cross_validate_on_sce",
                            help="cross validate on sce")
        parser.set_defaults(cross_validate_on_sce=False)
        parser.add_argument("-tn", action="store_true", dest="train_on_nll",
                            help="train on nll")
        parser.set_defaults(train_on_nll=False)
        parser.add_argument("-ta", action="store_true", dest="train_all",
                            help="train on ece and give all the metrics")
        parser.set_defaults(train_all=False)
        parser.add_argument("--max-dev-epoch", type=int, default=9, dest="max_dev_epoch",  
                            help='epoch to load')
        parser.add_argument("--temp", type=float, default=1.0, dest="temp", help="Optimal temperature")
        parser.add_argument("-log", action="store_true", dest="log",
                            help="whether to log")
        parser.set_defaults(log=False)



                            # untest on fine_grain yet.
        cuda_parser = parser.add_mutually_exclusive_group(required=False)
        cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
        cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
        cuda_parser.add_argument('--lower', dest='cuda', action='store_true')
        parser.set_defaults(cuda=True)
        parser.set_defaults(lower=True)

        args = parser.parse_args()
        return args
