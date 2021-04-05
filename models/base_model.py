# Python Package
import os
import logging
from copy import deepcopy
from collections import OrderedDict

# PyTorch Package
from torch.nn.parallel import DataParallel, DistributedDataParallel

# Original Package
from models.lr_scheduler import lr_scheduler as lr_scheduler

"""
This file normally contains the following features:
    - load / save / print network
    - the setting of learning rate, i.e. update lr / setup lr_scheduler
"""

class BaseModel():
    """ 
    Base model. 
    
    Args:
        logger: Responsible for displaying console output and runtime errors.
    """
    
    def __init__(self, logger):
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.logger = logger
        self.is_train = opt['is_train']

    ########## Basic Feature ##########
    def model_to_devices(self, net):
        """
        Model to device.

        Args:
            net (nn.Module)
        """
        # Single GPU
        net = net.to(self.device)
        
        # Multi GPU
        if self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        elif self.opt['dist']:
            # Distributed DataParallel
            pass
        return net

    def get_bare_model(self, net):
        """
        Get bare model, 
        especially under wrapping with DistributedDataParallel or DataParallel.
        """

        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """
        Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether stirctly loaded.
            param_key (str): The parameter key of loaded network. 
                             If set to None, use the root 'path'.
                             Default: 'params'.
        """
        net = self.get_bare_model(net)
        self.logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}')
        # Load all tensors onto the CPU, using a function
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)
        

    def save_networks(self, net, net_label, cur_iter, param_key='params'):
        """
        Save networks.

        Args:
            Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            cur_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """

        if cur_iter != -1:
            cur_iter = 'latest'
        save_filename = f'{net_label}_{cur_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key),
                'The length of net and param_key should be the same.'
        
        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                # remove unnecessary 'module.'
                if key.startswith('module.'):    # .startswith: search a designeated string
                    key = key[7:]
                state_dict[key] = param.cpu()
            state_dict[param_key_] = state_dict
        
        torch.save(save_dict, save_path)

    def save_training_state(self, epoch, cur_iter):
        """
        Save training states during training, which will be used for resuming.

        Args:
            epoch (int): Current epoch.
            cur_iter (int): Current iteration.
        """
        state = {
            'epoch': epoch,
            'iter': cur_iter,
            'optimizers': [],
            'schedulers': []
        }
        for optimizer in self.optimizers:
            state['optimizers'].append(optimizer.state_dict())
        for scheduler in self.schedulers:
            state['schedulers'].append(scheduler.state_dict())
        save_filename = f'{cur_iter}.state'
        save_path = os.path.join(self.opt['path']['training_states'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        """
        Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lenghts of schedulers'
        for i, optimizer in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(optimizer)
        for i, scheduler in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(scheduler)

    def print_network(self, net):
        """
        Print the str and parameter number of a network.
        """

        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (f'{net.__class__.__name__} - '
                           f'{net.module.__class__.__name__}')
        else:
            net_cls_str = f'{net.__class__.__name__}'
        
        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        self.logger.info(
            f'Networks: {net._cls_str}, with parameters: {net_params:,d}')
        self.logger.info(net_str)

    def _print_different_keys_loading(self, cur_net, load_net, strict=True):
        """
        Print keys with different name or different size when loading.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).
            
        Args:
            cur_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        cur_net = self.get_bare_model(cur_net)
        cur_net = cur_net.state_dict()
        cur_net_keys = set(cur_net.keys())
        load_net_keys = set(load_net.keys())

        if cur_net_keys != load_net_keys:
            self.logger.warning('Current net - loaded net:')
            for v in sorted(list(cur_net_keys - load_net_keys)):
                self.logger.warning(f' {v}')
            self.logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - cur_net_keys)):
                self.logger.warning(f'  {v}')
        
        # check the size for the same keys
        if not strict:
            common_keys = cur_net_keys & load_net_keys
            for k in common_keys:
                if cur_net[k].size() != load_net[k].size():
                    self.logger.warning(
                        f'Size different, ignore [{k}]: cur_net: '
                        f'{cur_net[k].shape}; load_net: {load_net[k].shape}')
                    )
                    load_net[k + '.ignore'] = load_net.pop(k)

    ########## Feature about Learning Rate ##########
    def _set_lr(self, lr_groups_l):
        """
        Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """
        Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append(
                [v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def get_current_learning_rate(self):
        return [
            param_group['lr']
            for param_group in self.optimizers[0].param_groups
        ]

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        """
        Update learning rate.

        Args:
            cur_iter (int): Current iteration.
            warmup_iter (int): Warmup iter number. -1 for no warmup.
                Default : -1.
        """
        if cur_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if cur_iter < warmup_iter:
            # get init_lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)
    
    def get_current_learning_rate(self):
        return [
            param_groups['lr']
            for param_group in self.optimizers[0].param_groups
        ]
    
    def setup_schedulers(self):
        """
        Set up schedulers.
        """
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR(optimizer,
                                                **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    

    

    