import torch
from torch import nn, optim
from torch.nn import functional as F
import sys


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature_exit1 = nn.Parameter(torch.ones(1) * 1.5)
        self.temperature_exit2 = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        return [ self.forward_exit(0, input),
                 self.forward_exit(1, input) ]

    def forward_exit(self, exit, input):
        logits = self.model.forward_exit(exit, input)
        return self.temperature_scale(exit, logits)

    def temperature_scale(self, exit, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        if exit == 0:
            temperature = self.temperature_exit1.unsqueeze(1).expand(logits.size(0), logits.size(1))
        elif exit == 1:
            temperature = self.temperature_exit2.unsqueeze(1).expand(logits.size(0), logits.size(1))

        return logits / temperature

    # This function probably should live outside of this class, but whatever

    def set_temperature(self, valid_loader):
        self.cuda()

        self.set_temperature_exit1(valid_loader)
        self.set_temperature_exit2(valid_loader)
        return self

    def set_temperature_exit1(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model.forward_exit(0, input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(f'Before temperature for 1 - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}')

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature_exit1], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(0, logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(0, logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(0, logits), labels).item()
        print(f'Optimal temperature for 1: {self.temperature_exit1.item():.3f}')
        print(f'After temperature for 1 - NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}')

        return self
    
    def set_temperature_exit2(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model.forward_exit(1, input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(f'Before temperature for 2 - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}')

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature_exit2], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(1, logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(1, logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(1, logits), labels).item()
        print(f'Optimal temperature for 2: {self.temperature_exit2.item():.3f}')
        print(f'After temperature for 2 - NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}')

        return self

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
