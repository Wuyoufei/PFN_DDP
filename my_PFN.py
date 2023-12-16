import torch
import torch.nn as nn
import numpy as np

class ParticleFlowNetwork(nn.Module):
    r"""Parameters
    ----------
    input_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    layer_params : list
        List of the feature size for each layer.
    """

    def __init__(self, input_dims, num_classes,
                 Phi_sizes=(100, 100, 128),
                 F_sizes=(100, 100, 100),
                 use_bn=True,
                 for_inference=False,
                 mask=True,
                 mask_val=0,
                 **kwargs):

        super(ParticleFlowNetwork, self).__init__(**kwargs)
        self.latent_dim=Phi_sizes[-1]
        self.input_dims=input_dims
        self.mask=mask
        self.mask_val=mask_val
        # input bn
        self.input_bn = nn.BatchNorm1d(input_dims) if use_bn else nn.Identity()
        # per-particle functions
        phi_layers = []
        for i in range(len(Phi_sizes)):
            phi_layers.append(nn.Sequential(
                nn.Conv1d(input_dims if i == 0 else Phi_sizes[i - 1], Phi_sizes[i], kernel_size=1),
                nn.BatchNorm1d(Phi_sizes[i]) if use_bn else nn.Identity(),
                nn.ReLU())
            )
        self.phi = nn.Sequential(*phi_layers)
        # global functions
        f_layers = []
        for i in range(len(F_sizes)):
            f_layers.append(nn.Sequential(
                nn.Linear(Phi_sizes[-1] if i == 0 else F_sizes[i - 1], F_sizes[i]),
                nn.ReLU())
            )
        f_layers.append(nn.Linear(F_sizes[-1], num_classes))
        if for_inference:
            f_layers.append(nn.Softmax(dim=1))
        self.fc = nn.Sequential(*f_layers)

    def forward(self, x):
        #device=x.device
        # x: the feature vector initally read from the data structure, in dimension (N, C, P)
        if self.mask:
            mask_matrix=(x.abs().sum(dim=1, keepdim=True) != self.mask_val)
        x = self.input_bn(x)
        x = self.phi(x)
        if self.mask:
            #x.data*=mask_matrix
            #x*=mask_matrix
            x=torch.mul(x,mask_matrix)
        x = x.sum(-1)
        return self.fc(x)
    
"""     def forward(self, x):
        device=x.device
        # x: the feature vector initally read from the data structure, in dimension (N, C, P)
        if self.mask:
            mask_matrix=self.get_mask_matrix(x).to(device=device)
        x = self.input_bn(x)
        x = self.phi(x)
        if self.mask:
            x=x.masked_fill(mask_matrix,0)
        x = x.sum(-1)
        return self.fc(x)
    
    def get_mask_matrix(self,x):
        mask=(x==self.mask_val).all(dim=1,keepdim=True)
        return mask """

"""     def forward(self, x):
        device=x.device
        # x: the feature vector initally read from the data structure, in dimension (N, C, P)
        if self.mask:
            mask_matrix=self.get_mask_matrix(x).to(device=device)
        x = self.input_bn(x)
        x = self.phi(x)
        if self.mask:
            x=x*mask_matrix
        x = x.sum(-1)
        return self.fc(x)
    
    def get_mask_matrix(self,x):
        mask_2d=torch.logical_not((x==self.mask_val).all(dim=1,keepdim=True))
        mask_3d=mask_2d.repeat(1,self.latent_dim,1).float()
        return mask_3d """
    
"""     def get_mask_matrix(self,x):
        num_events=x.shape[0]
        num_particles=x.shape[2]
        judge=np.ones_like(self.input_dims)*self.mask_val
        mask_=[]
        if x.device.type!='cpu':
            x_np=x.cpu().detach().numpy()
        else:
            x_np=x.detach().numpy()
        x_np=x_np.reshape(num_events,num_particles,-1)
        for i in range(num_events):
            for j in range(num_particles):
                if (x_np[i,j,:]==judge).all():
                    mask_.append(0)
                else:
                    mask_.append(1)
        mask_=np.array(mask_).reshape(num_events,num_particles)
        return mask_ """





def get_model(data_config, **kwargs):
    Phi_sizes = (128, 128, 128)
    F_sizes = (128, 128, 128)
    input_dims = len(data_config.input_dicts['pf_features'])
    num_classes = len(data_config.label_value)
    model = ParticleFlowNetwork(input_dims, num_classes, Phi_sizes=Phi_sizes,
                                F_sizes=F_sizes, use_bn=kwargs.get('use_bn', False))

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
