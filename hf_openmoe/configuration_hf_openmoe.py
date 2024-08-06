from transformers import LlamaConfig

__all__ = ["HFOpenMoeConfig"]


class HFOpenMoeConfig(LlamaConfig):
    model_type = "openmoe"
    def __init__(
            self,
            num_experts: int = 32,
            moe_layer_interval: int = 6,
            router_topk: int = 2,
            router_capacity_factor_train: float = 1.25,
            router_capacity_factor_eval: float = 2.0,
            router_min_capacity: int = 4,
            router_noisy_policy: str = None,
            router_drop_tks: bool = True,
            router_aux_loss_factor: float = 0.01,
            router_z_loss_factor: float = 0.0001,
            mlp_gated: bool = True,
            label_smoothing: float = 0.001,
            z_loss_factor: float = 0.01,
            enable_load_balance: bool = False,
            load_balance_tolerance: float = 0.1,
            load_balance_beam_width: int = 8,
            load_balance_group_swap_factor: float = 0.4,
            enable_kernel: bool = False,
            enable_comm_overlap: bool = False,
            enable_hierarchical_alltoall: bool = False,
            **kwargs
    ):
        self.num_experts = num_experts
        self.moe_layer_interval = moe_layer_interval
        self.router_topk = router_topk
        self.router_capacity_factor_train = router_capacity_factor_train
        self.router_capacity_factor_eval = router_capacity_factor_eval
        self.router_min_capacity = router_min_capacity
        self.router_noisy_policy = router_noisy_policy
        self.router_drop_tks = router_drop_tks
        self.router_aux_loss_factor = router_aux_loss_factor
        self.router_z_loss_factor = router_z_loss_factor
        self.mlp_gated = mlp_gated
        self.label_smoothing = label_smoothing
        self.z_loss_factor = z_loss_factor
        self.enable_load_balance = enable_load_balance
        self.load_balance_tolerance = load_balance_tolerance
        self.load_balance_beam_width = load_balance_beam_width
        self.load_balance_group_swap_factor = load_balance_group_swap_factor
        self.enable_kernel = enable_kernel
        self.enable_comm_overlap = enable_comm_overlap
        self.enable_hierarchical_alltoall = enable_hierarchical_alltoall

        super().__init__(**kwargs)
