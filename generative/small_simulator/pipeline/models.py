import numpy as np
import torch
import torch.nn as nn

try:
    import joblib
except Exception:
    joblib = None



class normalizer(nn.Module):
    def __init__(self, train_loader=None, input_size=10, output_size=3, inputs=list(), log=False):
        """
        A normalizer class to normalize and inverse normalize input and output tensors. 
        It calculates mean and standard deviation from the training data.
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            input_size (int): Size of the input features.
            output_size (int): Size of the output features.
            inputs (list): List of input feature names (for reference).
        """
        super(normalizer, self).__init__()
        if len(inputs)!=0:
            input_size = len(inputs)
            self.inputs = inputs

        self.register_buffer('mean_x', torch.zeros(input_size, dtype=torch.float32))
        self.register_buffer('std_x', torch.zeros(input_size, dtype=torch.float32))

        self.register_buffer('q9_b', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('q9_c', torch.zeros(1, dtype=torch.float32))

        self.register_buffer('mean_y', torch.zeros(output_size, dtype=torch.float32))
        self.register_buffer('std_y', torch.zeros(output_size, dtype=torch.float32))

        self.log = log
        if self.log:
            raise ValueError("Log output normalization is currently disabled.")
        if train_loader is not None:
            self.calculate_stats(train_loader)
    
    def log_transform(self, y):
        y[..., 0] = y[..., 0]
        y[..., 1] = torch.log(-y[..., 1]-self.q9_b)
        y[..., 2] = torch.log(-y[..., 2]-self.q9_c)
        return y

    def inverse_log_transform(self, y):
        y[..., 0] = y[..., 0]
        y[..., 1] = -torch.exp(y[..., 1]) - self.q9_b
        y[..., 2] = -torch.exp(y[..., 2]) - self.q9_c
        return y
    
    def calculate_stats(self, train_loader):
        all_data_x = torch.clone(train_loader.dataset.tensors[0])
        all_data_y = torch.clone(train_loader.dataset.tensors[1])
        
        if hasattr(self, 'inputs'):
            all_data_x = self.input_transform(all_data_x)
        self.mean_x = all_data_x.mean(dim=0)
        self.std_x = all_data_x.std(dim=0)
        self.std_x[self.std_x == 0] = 1


        if self.log:
            self.q9_b = all_data_y[:,1].quantile(0.9).reshape(1)
            self.q9_c = all_data_y[:,2].quantile(0.9).reshape(1)
            all_data_y = self.log_transform(all_data_y)

        self.mean_y = all_data_y.mean(dim=0)
        self.std_y = all_data_y.std(dim=0)
        self.std_y[self.std_y == 0] = 1
    
    def input_transform(self, x):
        for i, name in enumerate(self.inputs):
            if name=='energy_primary':
                x[:,i] = torch.log(x[:,i]/1e9)
            elif name=='omega':
                x[:,i] = torch.sqrt(x[:,i])
            elif name=="xmax_pos_z":
                x[:,i] = torch.square(x[:,i]/1000)
            elif name=="zenith":
                x[:,i] = torch.log(torch.cos(x[:,i]))

        return x
    
    def inverse_input_transform(self, x):
        for i, name in enumerate(self.inputs):
            if name=='energy_primary':
                x[:,i] = torch.exp(x[:,i])*1e9
            elif name=='omega':
                x[:,i] = torch.square(x[:,i])
            elif name=="xmax_pos_z":
                x[:,i] = torch.sqrt(x[:,i])*1000
            elif name=="zenith":
                x[:,i] = torch.acos(torch.exp(x[:,i]))
        return x
    
    def forward(self, vec, outputs=False):
        """
        Normalize the input tensor vec.
        Args:
            vec (torch.Tensor): Input tensor of shape (batch_size, input_size).
            outputs (bool): If True, normalize outputs instead of inputs.
        Returns:
            torch.Tensor: Normalized tensor.
        """
        if outputs:
            if (self.std_y == 0).all():
                raise ValueError("Normalizer has not been initialized with training data for outputs.")
            if self.log:
                vec = self.log_transform(vec)
            vec = (vec - self.mean_y) / self.std_y
            return vec
        

        else:
            if (self.std_x == 0).all():
                raise ValueError("Normalizer has not been initialized with training data.")
            if hasattr(self, 'inputs'):
                vec = self.input_transform(vec)
            return (vec - self.mean_x) / self.std_x

    def inverse(self, x, outputs=False):
        """
        Inverse normalize the input tensor x.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            outputs (bool): If True, inverse normalize outputs instead of inputs.
        Returns:
            torch.Tensor: Inverse normalized tensor.
        """
        if outputs:
            if (self.std_y == 0).all():
                print(self.std_y)
                raise ValueError("Normalizer has not been initialized with training data for outputs.")
            x = x * self.std_y + self.mean_y
            if self.log:
                x = self.inverse_log_transform(x)
            return x
        else:
            if (self.std_x == 0).all():
                raise ValueError("Normalizer has not been initialized with training data.")
            x = x * self.std_x + self.mean_x
            if hasattr(self, 'inputs'):
                x = self.inverse_input_transform(x)
            return x



def _get_activation(name):
    """Return an activation module from its name."""
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'silu' or name == 'swish':
        return nn.SiLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {name}. "
                         f"Choose from: relu, leaky_relu, gelu, silu, sigmoid, tanh.")


class LearnedWeightMSELoss(nn.Module):
    """
    Multi-output MSE with one learned weight per output (homoscedastic).

    Learns log_var_i = log(sigma_i^2) for each output i.  The loss is:

        L = sum_i [ 0.5 * exp(-log_var_i) * MSE_i  +  0.5 * log_var_i ]

    * exp(-log_var_i) = 1/sigma_i^2 acts as the effective weight.
    * The 0.5*log_var_i term prevents the trivial sigma -> inf solution.
    * Outputs that are harder to fit get larger sigma => smaller weight.

    Parameters
    ----------
    n_outputs : int
        Number of output features.
    """

    def __init__(self, n_outputs: int):
        super().__init__()
        # initialise log_var to 0  =>  sigma^2 = 1  =>  equal weights
        self.log_var = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred, target : (batch, n_outputs)

        Returns
        -------
        Scalar loss.
        """
        per_output_mse = (pred - target).pow(2).mean(dim=0)       # (n_outputs,)
        # 0.5 * (mse_i / sigma_i^2 + log(sigma_i^2))
        loss = 0.5 * (torch.exp(-self.log_var) * per_output_mse + self.log_var)
        return loss.sum()

    def effective_weights(self) -> torch.Tensor:
        """Return the current effective weight per output: 1 / sigma_i^2."""
        return torch.exp(-self.log_var).detach()

    def extra_repr(self) -> str:
        with torch.no_grad():
            w = self.effective_weights()
            parts = [f"{v:.4f}" for v in w]
        return f"weights=[{', '.join(parts)}]"


class HeteroscedasticNLLLoss(nn.Module):
    """
    Heteroskedastic Gaussian negative log-likelihood loss.

    Computes per-sample, per-output NLL. Expects predictions and an
    optional log-variance tensor (s = log sigma^2). If log_var is None,
    it behaves like standard MSE (for compatibility).

    Forward signature:
        loss = HeteroscedasticNLLLoss()(pred, target, log_var=None)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, log_var: torch.Tensor = None) -> torch.Tensor:
        """Compute scalar loss.

        pred, target: (batch, n_outputs)
        log_var: (batch, n_outputs) or (batch,)
        """
        if log_var is None:
            # fallback to mean squared error
            return ((pred - target).pow(2)).mean()

        # Ensure shapes align: allow log_var to be (batch, n_outputs) or (n_outputs,) broadcast
        # Numerically stable formulation: 0.5 * (exp(-s) * mse + s)
        s = log_var
        # clamp s to avoid extreme weights
        s = torch.clamp(s, min=torch.log(torch.tensor(self.eps)))

        per_sample_per_output = 0.5 * (torch.exp(-s) * (pred - target).pow(2) + s)
        return per_sample_per_output.mean()


class MLP_metamodel(nn.Module):
    def __init__(self, 
                 inputs=list(), 
                 var_head=False,
                 n_layers=7, 
                 skip_connection=0,
                 hidden_size=32,
                 activation='relu',
                 dropout=0.0,
                 input_size=None,
                 log_out=False,
                 output_size=3):
        super().__init__()
        """
        A simple MLP model with residual connections, LayerNorm and dropout.
        Args:
            inputs (list): List of input feature names.
            n_layers (int): Number of hidden layers.
            skip_connection (int): Add residual every N layers (0 = disabled).
            hidden_size (int): Width of hidden layers.
            activation (str): Activation name (relu, leaky_relu, gelu, silu, sigmoid, tanh).
            dropout (float): Dropout probability (0.0 = no dropout).
            input_size (int): Explicit input size (overridden by len(inputs) if inputs given).
            log_out (bool): Whether to apply log transform on outputs in normaliser.
            output_size (int): Number of output features.
        """
        self.output_size = output_size
        # whether to add an extra head that predicts log-variance per output
        self.var_head_enabled = var_head
        if len(inputs) != 0:
            input_size = len(inputs)
            self.inputs = inputs

        self.input_size = input_size
        # Define the layers of the MLP
        self.normalizer = normalizer(None, input_size=input_size, output_size=self.output_size, inputs=inputs, log=log_out)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = _get_activation(activation)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.hidden = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        self.hidden_act = _get_activation(activation)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        for _ in range(n_layers):
            self.hidden.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_norms.append(nn.LayerNorm(hidden_size))

        self.fout = nn.Linear(hidden_size, self.output_size)
        if self.var_head_enabled:
            # predicts log(sigma^2) per output
            self.var_head = nn.Linear(hidden_size, self.output_size)
        self.register_buffer('skip_connection', torch.tensor(skip_connection))
        # Weight initialisation
        self._init_weights(activation)

    def _init_weights(self, activation):
        """Apply proper weight initialisation depending on activation."""
        act = activation.lower()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if act in ('relu', 'leaky_relu'):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                elif act in ('gelu', 'silu', 'swish'):
                    # GELU/SiLU are close to linear near 0 — Xavier is a good default
                    nn.init.xavier_normal_(m.weight)
                elif act in ('sigmoid', 'tanh'):
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def initialize_normalizer(self, train_loader):
        """
        Initialize the normalizer with training data statistics.
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
        """
        self.normalizer.calculate_stats(train_loader)

        # var_head initialization: either empirical (data-driven) or constant.
        if getattr(self, 'var_head_enabled', False):
            # Decide whether to use empirical initialization based on a model
            # attribute set from the config; default is False to preserve
            # prior behavior.
            use_empirical = getattr(self, 'var_head_empirical_init', False)
            if use_empirical:
                try:
                    # Extract all targets from the train_loader's dataset tensors
                    all_targets = torch.clone(train_loader.dataset.tensors[1])
                    # Normalize targets using the same normalizer we just computed
                    targets_norm = self.normalizer(all_targets, outputs=True)
                    # empirical variance per output (use population variance)
                    var = targets_norm.var(dim=0, unbiased=False)
                    # avoid zeros / negatives
                    var = var.clamp(min=1e-6)
                    # set bias to log(var) so exp(bias) == var
                    with torch.no_grad():
                        if hasattr(self, 'var_head'):
                            # ensure shapes align
                            if self.var_head.bias.numel() == var.numel():
                                self.var_head.bias.copy_(torch.log(var))
                            else:
                                # fallback to zeros -> log(1)=0
                                self.var_head.bias.fill_(0.0)
                            if hasattr(self.var_head, 'weight'):
                                nn.init.zeros_(self.var_head.weight)
                except Exception:
                    # Best-effort: if anything goes wrong, default to log(1)=0
                    try:
                        with torch.no_grad():
                            if hasattr(self, 'var_head'):
                                self.var_head.bias.fill_(0.0)
                                if hasattr(self.var_head, 'weight'):
                                    nn.init.zeros_(self.var_head.weight)
                    except Exception:
                        pass
            else:
                try:
                    with torch.no_grad():
                        # set bias to 0 and weights to zero so initial var predictions are 1
                        self.var_head.bias.fill_(0.0)
                        if hasattr(self.var_head, 'weight'):
                            nn.init.zeros_(self.var_head.weight)
                except Exception:
                    # don't crash if model var_head shape mismatches or running on CPU/GPU timing
                    pass
        # # If a variance head is enabled, initialize its bias so that the
        # # initial predicted variance matches the empirical variance of the
        # # normalized targets (stable starting point). We also zero the
        # # weights so the initial prediction is constant per-output.
        # if getattr(self, 'var_head_enabled', False):
        #     try:
        #         # Extract all targets from the train_loader's dataset tensors
        #         all_targets = torch.clone(train_loader.dataset.tensors[1])
        #         # Normalize targets using the same normalizer we just computed
        #         targets_norm = self.normalizer(all_targets, outputs=True)
        #         # empirical variance per output (use population variance)
        #         var = targets_norm.var(dim=0, unbiased=False)
        #         # avoid zeros / negatives
        #         var = var.clamp(min=1e-6)
        #         # set bias to log(var) so exp(bias) == var
        #         with torch.no_grad():
        #             if hasattr(self, 'var_head'):
        #                 # ensure shapes align
        #                 if self.var_head.bias.numel() == var.numel():
        #                     self.var_head.bias.copy_(torch.log(var))
        #                 else:
        #                     # fallback to zeros -> log(1)=0
        #                     self.var_head.bias.fill_(0.0)
        #                 if hasattr(self.var_head, 'weight'):
        #                     nn.init.zeros_(self.var_head.weight)
        #     except Exception:
        #         # Best-effort: if anything goes wrong, default to log(1)=0
        #         try:
        #             with torch.no_grad():
        #                 if hasattr(self, 'var_head'):
        #                     self.var_head.bias.fill_(0.0)
        #                     if hasattr(self.var_head, 'weight'):
        #                         nn.init.zeros_(self.var_head.weight)
        #         except Exception:
        #             pass

    def forward(self, x):
        """
        Forward pass of the MLP model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        x = self.normalizer(x)        
        xh = self.dropout(self.act1(self.ln1(self.fc1(x))))
        for i, (layer, ln) in enumerate(zip(self.hidden, self.hidden_norms)):
            out = self.dropout(self.hidden_act(ln(layer(xh))))
            if self.skip_connection and (i + 1) % self.skip_connection == 0:
                xh = out + xh
            else:
                xh = out
        x = self.fout(xh)
        # If var head enabled, also predict log-variance (no activation)
        if getattr(self, 'var_head_enabled', False):
            s = self.var_head(xh)
            return x, s
        return x


class MLP_metamodel_betterskip(nn.Module):
    def __init__(self, 
                 inputs=list(), 
                 var_head=False,
                 n_layers=7, 
                 skip_connection=0,
                 hidden_size=32,
                 activation='relu',
                 dropout=0.0,
                 input_size=None,
                 log_out=False,
                 output_size=3):
        super().__init__()
        """
        A simple MLP model with residual connections, LayerNorm and dropout.
        Args:
            inputs (list): List of input feature names.
            n_layers (int): Number of hidden layers.
            skip_connection (int): Add residual every N layers (0 = disabled).
            hidden_size (int): Width of hidden layers.
            activation (str): Activation name (relu, leaky_relu, gelu, silu, sigmoid, tanh).
            dropout (float): Dropout probability (0.0 = no dropout).
            input_size (int): Explicit input size (overridden by len(inputs) if inputs given).
            log_out (bool): Whether to apply log transform on outputs in normaliser.
            output_size (int): Number of output features.
        """
        self.output_size = output_size
        # whether to add an extra head that predicts log-variance per output
        self.var_head_enabled = var_head
        if len(inputs) != 0:
            input_size = len(inputs)
            self.inputs = inputs

        self.input_size = input_size
        # Define the layers of the MLP
        self.normalizer = normalizer(None, input_size=input_size, output_size=self.output_size, inputs=inputs, log=log_out)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = _get_activation(activation)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.hidden = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        self.hidden_act = _get_activation(activation)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        for _ in range(n_layers):
            self.hidden.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_norms.append(nn.LayerNorm(hidden_size))

        self.fout = nn.Linear(hidden_size, self.output_size)
        if self.var_head_enabled:
            # predicts log(sigma^2) per output
            self.var_head = nn.Linear(hidden_size, self.output_size)
        self.register_buffer('skip_connection', torch.tensor(skip_connection))
        self.skip_tails = [i for i in range(n_layers) if skip_connection > 0 and (i - 1) % skip_connection == 0]
        self.skip_heads = [i + skip_connection for i in self.skip_tails if i + skip_connection < n_layers]
        if len(self.skip_heads) == 0 and skip_connection > 0:
            raise ValueError("skip_connection is set to a value that results in no skip connections. ")
        # After one hidden layer, add a skip connection every N layers that connects N layers apart (if skip_connection > 0). 
        # For example, if skip_connection=2, add a skip connection from layer 1 to layer 3, from layer 3 to layer 5, etc. 
        # Weight initialisation
        self._init_weights(activation)

    def _init_weights(self, activation):
        """Apply proper weight initialisation depending on activation."""
        act = activation.lower()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if act in ('relu', 'leaky_relu'):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                elif act in ('gelu', 'silu', 'swish'):
                    # GELU/SiLU are close to linear near 0 — Xavier is a good default
                    nn.init.xavier_normal_(m.weight)
                elif act in ('sigmoid', 'tanh'):
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def initialize_normalizer(self, train_loader):
        """
        Initialize the normalizer with training data statistics.
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
        """
        self.normalizer.calculate_stats(train_loader)

        # var_head initialization: either empirical (data-driven) or constant.
        if getattr(self, 'var_head_enabled', False):
            # Decide whether to use empirical initialization based on a model
            # attribute set from the config; default is False to preserve
            # prior behavior.
            use_empirical = getattr(self, 'var_head_empirical_init', False)
            if use_empirical:
                try:
                    # Extract all targets from the train_loader's dataset tensors
                    all_targets = torch.clone(train_loader.dataset.tensors[1])
                    # Normalize targets using the same normalizer we just computed
                    targets_norm = self.normalizer(all_targets, outputs=True)
                    # empirical variance per output (use population variance)
                    var = targets_norm.var(dim=0, unbiased=False)
                    # avoid zeros / negatives
                    var = var.clamp(min=1e-6)
                    # set bias to log(var) so exp(bias) == var
                    with torch.no_grad():
                        if hasattr(self, 'var_head'):
                            # ensure shapes align
                            if self.var_head.bias.numel() == var.numel():
                                self.var_head.bias.copy_(torch.log(var))
                            else:
                                # fallback to zeros -> log(1)=0
                                self.var_head.bias.fill_(0.0)
                            if hasattr(self.var_head, 'weight'):
                                nn.init.zeros_(self.var_head.weight)
                except Exception:
                    # Best-effort: if anything goes wrong, default to log(1)=0
                    try:
                        with torch.no_grad():
                            if hasattr(self, 'var_head'):
                                self.var_head.bias.fill_(0.0)
                                if hasattr(self.var_head, 'weight'):
                                    nn.init.zeros_(self.var_head.weight)
                    except Exception:
                        pass
            else:
                try:
                    with torch.no_grad():
                        # set bias to 0 and weights to zero so initial var predictions are 1
                        self.var_head.bias.fill_(0.0)
                        if hasattr(self.var_head, 'weight'):
                            nn.init.zeros_(self.var_head.weight)
                except Exception:
                    # don't crash if model var_head shape mismatches or running on CPU/GPU timing
                    pass

    def forward(self, x):
        """
        Forward pass of the MLP model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        x = self.normalizer(x)        
        xh = self.dropout(self.act1(self.ln1(self.fc1(x))))
        for i, (layer, ln) in enumerate(zip(self.hidden, self.hidden_norms)):
            out = self.dropout(self.hidden_act(ln(layer(xh))))
            if self.skip_connection and i in self.skip_tails:
                skip_value = out
            if self.skip_connection and i in self.skip_heads:
                # add skip connection from the corresponding tail layer
                out = out + skip_value
                skip_value = out
            xh = out


            # Badly implemented skip connection every N layers (if enabled in config).
            # if self.skip_connection and (i + 1) % self.skip_connection == 0:
            #     xh = out + xh
            # else:
            #     xh = out
        x = self.fout(xh)
        # If var head enabled, also predict log-variance (no activation)
        if getattr(self, 'var_head_enabled', False):
            s = self.var_head(xh)
            return x, s
        return x


class MLPClassifierGated(nn.Module):
    """
    Wrap an MLP_metamodel with a fast classifier to down-weight outputs
    when the classifier predicts low probability of valid (non-NaN) output.

    The classifier is expected to implement predict_proba and return the
    positive-class probability at index 1. HistGradientBoostingClassifier
    is supported out of the box.
    """

    def __init__(
        self,
        mlp_model: nn.Module,
        classifier=None,
        classifier_path: str = None,
        penalty_scale: float = 1.0,
        penalty_power: float = 1.0,
        proba_eps: float = 1e-4,
    ):
        super().__init__()
        self.mlp_model = mlp_model
        self.normalizer = getattr(mlp_model, "normalizer", None)
        self.output_size = getattr(mlp_model, "output_size", None)

        if classifier is None and classifier_path is not None:
            if joblib is None:
                raise ImportError("joblib is required to load classifier_path")
            classifier = joblib.load(classifier_path)
        self.classifier = classifier

        self.penalty_scale = penalty_scale
        self.penalty_power = penalty_power
        self.proba_eps = 0
        self.device = next(mlp_model.parameters()).device if mlp_model is not None else torch.device('cpu')

    def forward(self, x: torch.Tensor):
        return self.mlp_model(x)

    def _predict_proba(self, params):
        if self.classifier is None:
            return None
        if isinstance(params, torch.Tensor):
            params_np = params.detach().cpu().numpy()
        else:
            params_np = np.asarray(params)

        proba = self.classifier.predict_proba(params_np)
        if proba.ndim == 1:
            pos = proba
        else:
            pos = proba[:, 1]
        return np.clip(pos, self.proba_eps, 1.0 - self.proba_eps)

    def postprocess_outputs(self, params, preds):
        """
        Apply a penalty to a,b,c when the classifier predicts low validity.
        This runs in output (physical) space after inverse normalization.
        """
        proba = self._predict_proba(params)
        if proba is None:
            return preds
        
        penalty = self.penalty_scale * (1.0 - proba) ** self.penalty_power*(proba < 0.2)

        print(penalty)
        preds = np.asarray(preds).copy()
        if preds.shape[1] >= 3:
            preds[:, 0:3] -= penalty[:, None]
        return preds
    