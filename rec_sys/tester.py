import torch
import wandb
from torch import nn
from torch.utils import data

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from rec_sys.rec_sys import RecSysMeTa
from utilities.eval import Evaluator
from utilities.utils import print_results


class Tester:

    def __init__(self, test_loader: data.DataLoader, conf, model_load_path: str):
        """
        Test the model
        :param test_loader: Test DataLoader (check music4all_data.Music4AllDataset for more info)
        :param conf: Experiment configuration parameters
        :param model_load_path: Path to load the model to test
        """

        self.test_loader = test_loader

        self.conf = conf
        self.ft_ext_param = conf.ft_ext_param
        self.model_load_path = model_load_path
        self.optim_param = conf.optim_param
        self.u_p_l1 = self.optim_param['u_p_l1']

        self.loss_func_name = conf.loss_func_name
        self.loss_func_aggr = conf.loss_func_aggr if 'loss_func_aggr' in conf else 'mean'

        self.device = conf.device

        self.model = self._build_model()

        print(f'Built Tester module \n'
              f'- loss_func_name: {self.loss_func_name} \n'
              f'- loss_func_aggr: {self.loss_func_aggr} \n'
              f'- device: {self.device} \n'
              f'- model_load_path: {self.model_load_path} \n')

    def _build_model(self):
        n_users = self.test_loader.dataset.n_users
        n_items = self.test_loader.dataset.n_items

        user_sim, item_sim, proto_weight = \
            FeatureExtractorFactory.create_meta_models(self.ft_ext_param, n_users, n_items)
        # Step 2 --- Building RecSys Module
        rec_sys = RecSysMeTa(user_sim, item_sim, proto_weight,
                             self.loss_func_name, self.loss_func_aggr)

        rec_sys.init_parameters()

        # Step 3 --- Loading
        params = torch.load(self.model_load_path, map_location=self.device)
        rec_sys.load_state_dict(params)
        rec_sys = rec_sys.to(self.device)
        print('Model Loaded')

        return rec_sys

    @torch.no_grad()
    def test(self):
        """
        Runs the evaluation procedure.

        """
        self.model.eval()
        print('Testing started')
        test_loss = 0
        uipc_mf_l1_eval_out = Evaluator(self.test_loader.dataset.n_users)
        for u_idxs, i_idxs, labels in self.test_loader:
            u_idxs = u_idxs.to('cuda')
            i_idxs = i_idxs.to('cuda')
            labels = labels.to(self.device)
            uipc_mf_l1_out, u_prefer_l1 = self.model(u_idxs, i_idxs)
            total_loss = self.model.loss_func(uipc_mf_l1_out, labels)+u_prefer_l1*self.u_p_l1
            test_loss += total_loss.item()

            uipc_mf_l1_out = nn.Sigmoid()(uipc_mf_l1_out)
            uipc_mf_l1_out = uipc_mf_l1_out.to('cpu')

            uipc_mf_l1_eval_out.eval_batch(uipc_mf_l1_out)

        test_loss /= len(self.test_loader)

        metrics_values = {**uipc_mf_l1_eval_out.get_results(result_type='test_uipc_mf_l1_'),
                          'test_loss': test_loss}

        print_results(metrics_values)

        try:
            wandb.log(metrics_values)
        except wandb.Error:
            print('Not logged to wandb!')

        return metrics_values

    @torch.no_grad()
    def get_test_logits(self):
        """
                Returns the Logits on the Test Dataset
        """
        self.model.eval()
        print('Testing started')

        uipc_mf_l1_eval_out = Evaluator(self.test_loader.dataset.n_users)

        for u_idxs, i_idxs, labels in self.test_loader:
            u_idxs = u_idxs.to(self.device)
            i_idxs = i_idxs.to('cuda')

            uipc_mf_l1_out, u_prefer_l1 = self.model(u_idxs, i_idxs)

            uipc_mf_l1_out = nn.Sigmoid()(uipc_mf_l1_out)
            uipc_mf_l1_out = uipc_mf_l1_out.to('cpu')

            uipc_mf_l1_eval_out.eval_batch(uipc_mf_l1_out, sum=False)

        results = {**uipc_mf_l1_eval_out.get_results(aggregated=False, result_type='test_uipc_mf_l1_')}

        return results
