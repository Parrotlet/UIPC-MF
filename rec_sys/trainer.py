import os

import torch
from ray import tune
from torch import nn
from torch.utils import data

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from rec_sys.rec_sys import RecSysMeTa
from utilities.consts import OPTIMIZING_METRIC, MAX_PATIENCE
from utilities.eval import Evaluator


class Trainer:

    def __init__(self, train_loader: data.DataLoader, val_loader: data.DataLoader, conf):
        """
        Train and Evaluate the model.
        :param train_loader: Training DataLoader (check music4all_data.Music4AllDataset for more info)
        :param val_loader: Validation DataLoader (check music4all_data.Music4AllDataset for more info)
        :param conf: Experiment configuration parameters
        """

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.conf = conf
        self.ft_ext_param = conf.ft_ext_param
        self.optim_param = conf.optim_param

        self.n_epochs = conf.n_epochs
        self.loss_func_name = conf.loss_func_name
        self.loss_func_aggr = conf.loss_func_aggr if 'loss_func_aggr' in conf else 'mean'

        self.device = conf.device

        self.optimizing_metric = OPTIMIZING_METRIC
        self.max_patience = MAX_PATIENCE
        self.u_p_l1 = self.optim_param['u_p_l1']

        self.model = self._build_model()
        self.optimizer = self._build_optimizer()

        # print(f'Built Trainer module \n'
        #       f'- n_epochs: {self.n_epochs} \n'
        #       f'- loss_func_name: {self.loss_func_name} \n'
        #       f'- loss_func_aggr: {self.loss_func_aggr} \n'
        #       f'- device: {self.device} \n'
        #       f'- optimizing_metric: {self.optimizing_metric} \n')

    def _build_model(self):
        # Step 1 --- Building User and Item Feature Extractors
        n_users = self.train_loader.dataset.n_users
        n_items = self.train_loader.dataset.n_items

        user_sim, item_sim, proto_weight = \
            FeatureExtractorFactory.create_meta_models(self.ft_ext_param, n_users, n_items)
        # Step 2 --- Building RecSys Module
        rec_sys = RecSysMeTa(user_sim, item_sim, proto_weight,
                             self.loss_func_name, self.loss_func_aggr)

        rec_sys.init_parameters()

        rec_sys = rec_sys.to(self.device)

        return rec_sys

    def _build_optimizer(self):
        self.lr = self.optim_param['lr'] if 'lr' in self.optim_param else 1e-3
        self.wd = self.optim_param['wd'] if 'wd' in self.optim_param else 1e-4

        optim_name = self.optim_param['optim']
        if optim_name == 'adam':
            optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif optim_name == 'adagrad':
            optim = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            raise ValueError('Optimizer not yet included')

        # print(f'Built Optimizer  \n'
        #       f'- name: {optim_name} \n'
        #       f'- lr: {self.lr} \n'
        #       f'- wd: {self.wd} \n')

        return optim

    def run(self):
        """
        Runs the Training procedure
        """

        metrics_values = 0
        print('Init')
        best_value = 0


        patience = 0
        curr_value = 0
        for epoch in range(self.n_epochs):

            if patience == self.max_patience:
                print('Max Patience reached, stopping.')
                break
            # if epoch > 0 and curr_value < 0.25:
            #     break

            self.model.train()

            epoch_train_loss = 0

            for u_idxs, i_idxs, labels in self.train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                u_idxs = u_idxs.to('cuda')
                i_idxs = i_idxs.to('cuda')

                labels = labels.to(self.device)
                uipc_mf_l1_out, u_prefer_l1 = self.model(u_idxs, i_idxs)

                total_loss = self.model.loss_func(uipc_mf_l1_out, labels)+u_prefer_l1*self.u_p_l1


                loss = total_loss.item()
                epoch_train_loss += loss
                total_loss.backward()
                self.optimizer.step()


            epoch_train_loss /= len(self.train_loader)
            print("Epoch {} - Epoch Avg Train Loss {:.3f} \n".format(epoch, epoch_train_loss))

            metrics_values = self.val()
            curr_value = metrics_values['val_uipc_mf_l1_' + self.optimizing_metric]
            print('Epoch {} - Avg Val Value {:.3f} \n'.format(epoch, curr_value),
                  f'- conf: {self.conf} \n')

            tune.report({**metrics_values, 'epoch_train_loss': epoch_train_loss})
            # del epoch_train_loss, metrics_values
            if curr_value > best_value:
                best_value = curr_value
                print('Epoch {} - New best model found (val value {:.3f}) \n'.format(epoch, curr_value))
                with tune.checkpoint_dir(0) as checkpoint_dir:
                    torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))

                patience = 0
            else:
                patience += 1
        # torch.cuda.empty_cache()

    @torch.no_grad()
    def val(self):
        """
        Runs the evaluation procedure.
        :return: A scalar float value, output of the validation (e.g. NDCG@10).
        """
        self.model.eval()
        print('Validation started')
        val_loss = 0
        uipc_mf_l1_eval_out = Evaluator(self.val_loader.dataset.n_users)
        for u_idxs, i_idxs, labels in self.val_loader:
            u_idxs = u_idxs.to('cuda')
            i_idxs = i_idxs.to('cuda')

            labels = labels.to(self.device)
            uipc_mf_l1_out,u_prefer_l1 = self.model(u_idxs, i_idxs)

            total_loss = self.model.loss_func(uipc_mf_l1_out, labels)+u_prefer_l1*self.u_p_l1
            loss = total_loss.item()
            val_loss += loss

            uipc_mf_l1_out = nn.Sigmoid()(uipc_mf_l1_out)
            uipc_mf_l1_out = uipc_mf_l1_out.to('cpu')

            uipc_mf_l1_eval_out.eval_batch(uipc_mf_l1_out)

        val_loss /= len(self.val_loader)
        metrics_values = {**uipc_mf_l1_eval_out.get_results(result_type='val_uipc_mf_l1_'),
                          'val_loss': val_loss}

        return metrics_values
