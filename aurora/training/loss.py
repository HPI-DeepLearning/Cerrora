import torch

from aurora import Batch

def compute_latitude_weights(phi_tensor):
    """
    compute_latitude_weights based on the given formula.
    https://ekapex.atlassian.net/wiki/spaces/SD/pages/168230913/CERRA+projection+and+weights

    Parameters:
    ----------
    phi_tensor : torch.Tensor
        A tensor of shape (1056, 1056) where each element represents a latitude φ in degrees.

    Returns:
    ----------
    k_neg2 : torch.Tensor
        A tensor of the same shape as phi_tensor, where each element is the computed k^(-2) value.
    """
    # 1. Define the reference latitude phi1 (in degrees)
    phi1_deg = 50.0

    # 2. Convert the reference latitude from degrees to radians
    phi1 = torch.deg2rad(torch.tensor(phi1_deg))

    # 3. Compute n = sin(phi1)
    n = torch.sin(phi1)

    # 4. Convert the input tensor from degrees to radians
    phi_rad = torch.deg2rad(phi_tensor)

    # 5. Compute k according to the formula:
    #    k = [cos(phi1) * tan^(n)(pi/4 + phi1/2)] / [cos(phi) * tan^(n)(pi/4 + phi/2)]
    numerator = torch.cos(phi1) * torch.pow(torch.tan((torch.pi / 4) + (phi1 / 2)), n)
    denominator = torch.cos(phi_rad) * torch.pow(torch.tan((torch.pi / 4) + (phi_rad / 2)), n)
    k = numerator / denominator

    # 6. Compute k^(-2)
    k_neg2 = torch.pow(k, -2)

    # 7. Normalize the weights to mean 1
    k_neg2 = k_neg2 / k_neg2.mean()

    return k_neg2

class AuroraMeanAbsoluteError(torch.nn.Module):

    def __init__(
        self,
        alpha=1 / 4,
        beta=1,
        variable_weights=None,
        dataset_weights=None,
        latitude_weights=None,
    ):
        super(AuroraMeanAbsoluteError, self).__init__()

        # define weights tensor
        self.variable_weights = variable_weights

        self.alpha = alpha
        self.beta = beta
        self.dataset_weights = dataset_weights if dataset_weights else {"default": 1.0}
        self.latitude_weights = latitude_weights


    def forward(self, pred: Batch, target: Batch, dataset_name: str = "default"):
        gamma = self.dataset_weights.get(dataset_name, 1.0)

        surf_var_names = pred.surf_vars.keys()
        atmos_var_names = pred.atmos_vars.keys()
        surf_weights = torch.tensor([self.variable_weights["surface"][name] for name in surf_var_names])
        atmos_weights = torch.tensor([self.variable_weights["atmospheric"][name] for name in atmos_var_names])

        # Surface loss calculation
        pred_surf = torch.stack([pred.surf_vars[var] for var in surf_var_names]).squeeze(2)
        target_surf = torch.stack([target.surf_vars[var] for var in surf_var_names]).squeeze(2)

        # surf_diffs is avg. over B x H × W
        surf_diffs = torch.abs(pred_surf - target_surf).mean(dim=1)
        surf_diffs = (surf_diffs * self.latitude_weights.to(surf_diffs.device)).mean(dim=(1, 2))
        surface_loss = (surf_diffs * surf_weights.to(surf_diffs.device)).sum()
        surface_loss *= self.alpha

        # Atmospheric loss calculation
        pred_atmos = torch.stack([pred.atmos_vars[var] for var in atmos_var_names]).squeeze(2)
        target_atmos = torch.stack([target.atmos_vars[var] for var in atmos_var_names]).squeeze(2)

        # atmos_diffs is avg. over B x C × H × W
        atmos_diffs = torch.abs(pred_atmos - target_atmos).mean(dim=(1, 2,))
        atmos_diffs = (atmos_diffs * self.latitude_weights.to(surf_diffs.device)).mean(dim=(1, 2))
        atmospheric_loss = (atmos_diffs * atmos_weights.to(atmos_diffs.device)).sum()
        atmospheric_loss *= self.beta

        # Total loss calculation
        total_vars = len(surf_var_names) + len(atmos_var_names)
        total_loss = (gamma / total_vars) * (surface_loss + atmospheric_loss)

        return total_loss
