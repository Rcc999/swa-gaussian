import abc
import collections
import copy
import enum
import math
import pathlib
import typing
import warnings

import numpy as np
import torch
import torch.optim
import torch.utils.data
import tqdm
from matplotlib import pyplot as plt

from util import paint_reliability_diagram, compute_cost, seed_setup, calculate_calibration_curve

ENABLE_EXTENDED_ANALYSIS = False
"""
Set `ENABLE_EXTENDED_ANALYSIS` to `True` in order to generate additional plots on validation data.
"""

USE_PRETRAINED_WEIGHTS = True
"""
If `USE_PRETRAINED_WEIGHTS` is `True`, then MAP inference uses provided pretrained weights.
You should not modify MAP training or the CNN architecture before passing the baseline.
If you set the constant to `False` (to further experiment),
this solution always performs MAP inference before running your SWAG implementation.
Note that MAP inference can take a long time.
"""


def main():
    raise RuntimeError(
        "This main() method is for illustrative purposes only"
        " and will NEVER be called when running your solution to generate your submission file!\n"
        "The checker always directly interacts with your SWAGPredictor class and evaluate_model method.\n"
        "You can remove this exception for local testing, but be aware that any changes to the main() method"
        " are ignored when generating your submission file."
    )

    data_location = pathlib.Path.cwd()
    model_location = pathlib.Path.cwd()
    output_location = pathlib.Path.cwd()

    # Load training data
    training_images = torch.from_numpy(np.load(data_location / "train_xs.npz")["train_xs"])
    training_metadata = np.load(data_location / "train_ys.npz")
    training_labels = torch.from_numpy(training_metadata["train_ys"])
    training_snow_labels = torch.from_numpy(training_metadata["train_is_snow"])
    training_cloud_labels = torch.from_numpy(training_metadata["train_is_cloud"])
    training_dataset = torch.utils.data.TensorDataset(training_images, training_snow_labels, training_cloud_labels, training_labels)

    # Load validation data
    validation_images = torch.from_numpy(np.load(data_location / "val_xs.npz")["val_xs"])
    validation_metadata = np.load(data_location / "val_ys.npz")
    validation_labels = torch.from_numpy(validation_metadata["val_ys"])
    validation_snow_labels = torch.from_numpy(validation_metadata["val_is_snow"])
    validation_cloud_labels = torch.from_numpy(validation_metadata["val_is_cloud"])
    validation_dataset = torch.utils.data.TensorDataset(validation_images, validation_snow_labels, validation_cloud_labels, validation_labels)

    # Fix all randomness
    seed_setup()

    # Build and run the actual solution
    swag_inference = SWAGPredictor(
        train_images=training_dataset.tensors[0],
        weights_dir=model_location,
    )
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=swag_inference.swag_batch_size,
        shuffle=True,
        num_workers=0,
    )
    swag_inference.run_training(training_loader)
    swag_inference.calibrate_confidence(validation_dataset)

    # fork_rng ensures that the evaluation does not change the rng state.
    # That way, you should get exactly the same results even if you remove evaluation
    # to save computational time when developing the task
    # (as long as you ONLY use torch randomness, and not e.g. random or numpy.random).
    with torch.random.fork_rng():
        evaluate_model(swag_inference, validation_dataset, ENABLE_EXTENDED_ANALYSIS, output_location)


class InferenceMode(enum.Enum):
    """
    Inference mode switch for your implementation.
    `MAP` simply predicts the most likely class using pretrained MAP weights.
    `SWAG_DIAGONAL` and `SWAG_FULL` correspond to SWAG-diagonal and the full SWAG method, respectively.
    """
    MAP = 0
    SWAG_DIAGONAL = 1
    SWAG_FULL = 2


class SWAGPredictor(object):
    """
    Your implementation of SWA-Gaussian.
    This class is used to run and evaluate your solution.
    You must preserve all methods and signatures of this class.
    However, you can add new methods if you want.

    We provide basic functionality and some helper methods.
    You can pass the baseline by only modifying methods marked with TODO.
    However, we encourage you to skim other methods in order to gain a better understanding of SWAG.
    """

    def __init__(
        self,
        train_images: torch.Tensor,
        weights_dir: pathlib.Path,
        inference_mode: InferenceMode = InferenceMode.SWAG_FULL,
        # TODO(2): optionally add/tweak hyperparameters
        swag_training_epochs: int = 30,
        swag_batch_size: int = 16,
        swag_lr: float = 0.02,
        swag_update_interval: int = 1,
        max_rank_deviation_matrix: int = 15,
        num_bma_samples: int = 30,
        swag_burn_in_epochs: int = 5,
        swag_updates_per_epoch: int = 4,
        swag_augment: bool = False,
        swag_cutmix_prob: float = 0.0,
        test_time_augmentation: str = "none",
    ):
        """
        :param train_images: Training images (for storage only)
        :param weights_dir: Path to directory containing pretrained MAP weights
        :param inference_mode: Control which inference mode (MAP, SWAG-diagonal, full SWAG) to use
        :param swag_training_epochs: Total number of gradient descent epochs for SWAG
        :param swag_batch_size: Batch size for SWAG gradient descent (used by the runner to build the training loader)
        :param swag_lr: Learning rate for SWAG gradient descent
        :param swag_update_interval: Frequency (in epochs) for updating SWAG statistics during gradient descent
        :param max_rank_deviation_matrix: Rank of deviation matrix for full SWAG
        :param num_bma_samples: Number of networks to sample for Bayesian model averaging during prediction
        :param swag_burn_in_epochs: Number of initial SWAG epochs during which no statistics are collected
        :param swag_updates_per_epoch: Number of (evenly spaced) SWAG snapshots per collecting epoch
        :param swag_augment: Apply a random rotation/flip (dihedral group D4) to every SWAG training image
        :param swag_cutmix_prob: Probability of applying CutMix to a SWAG training batch
        :param test_time_augmentation: "none" (original images only), "cycle" (posterior sample s predicts
            on dihedral view s mod 8, same cost as "none"), or "full" (every sample predicts on all 8 views)
        """
        assert test_time_augmentation in ("none", "cycle", "full")

        self.weights_dir = weights_dir
        self.inference_mode = inference_mode
        self.swag_training_epochs = swag_training_epochs
        self.swag_batch_size = swag_batch_size
        self.swag_lr = swag_lr
        self.swag_update_interval = swag_update_interval
        self.max_rank_deviation_matrix = max_rank_deviation_matrix
        self.num_bma_samples = num_bma_samples
        self.swag_burn_in_epochs = swag_burn_in_epochs
        self.swag_updates_per_epoch = swag_updates_per_epoch
        self.swag_augment = swag_augment
        self.swag_cutmix_prob = swag_cutmix_prob
        self.test_time_augmentation = test_time_augmentation

        # Network used to perform SWAG.
        # Note that all operations in this class modify this network IN-PLACE!
        self.network = CNN(in_channels=3, out_classes=6)

        # Store training dataset to recalculate batch normalization statistics during SWAG inference
        self.training_dataset = torch.utils.data.TensorDataset(train_images)

        # SWAG-diagonal: running first and second moments of the weights, and the number of snapshots
        self._swag_num_snapshots = 0
        self._swag_mean = self._create_weight_copy()
        self._swag_sq_mean = self._create_weight_copy()

        # Full SWAG: the last K deviations (theta_i - running mean), one per-layer dict per snapshot
        self._swag_deviations = collections.deque(maxlen=self.max_rank_deviation_matrix)

        # Calibration, prediction, and other attributes
        self._calibration_threshold = None
        # Temperature applied to every sampled network's logits before the softmax (fit on validation data)
        self._temperature = 1.0
        # Posterior samples (state dicts incl. batchnorm statistics), drawn once and reused for all predictions
        self._posterior_samples = None

    def update_swag_statistics(self) -> None:
        """
        Update SWAG statistics with the current weights of self.network.
        """

        # Create a copy of the current network weights
        copied_params = {name: param.detach() for name, param in self.network.named_parameters()}

        # SWAG-diagonal
        n = self._swag_num_snapshots
        for name, param in copied_params.items():
            self._swag_mean[name] = (n * self._swag_mean[name] + param) / (n + 1)
            self._swag_sq_mean[name] = (n * self._swag_sq_mean[name] + param ** 2) / (n + 1)
        self._swag_num_snapshots += 1

        # Full SWAG
        if self.inference_mode == InferenceMode.SWAG_FULL:
            # Deviation from the running mean that already includes the current snapshot (as in Maddox et al.)
            self._swag_deviations.append(
                {name: param - self._swag_mean[name] for name, param in copied_params.items()}
            )

    def fit_swag_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        Fit SWAG on top of the pretrained network self.network.
        This method should perform gradient descent with occasional SWAG updates
        by calling self.update_swag_statistics().
        """

        # We use SGD with momentum and weight decay to perform SWA.
        # See the paper on how weight decay corresponds to a type of prior.
        # Feel free to play around with optimization hyperparameters.
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.swag_lr,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )
        # TODO(2): Update SWAGScheduler instantiation if you decided to implement a custom schedule.
        #  By default, this scheduler just keeps the initial learning rate given to `optimizer`.
        lr_scheduler = SWAGScheduler(
            optimizer,
            epochs=self.swag_training_epochs,
            steps_per_epoch=len(loader),
        )

        # Reset all SWAG statistics; the pretrained MAP weights themselves are not a snapshot
        self._swag_num_snapshots = 0
        self._swag_mean = self._create_weight_copy()
        self._swag_sq_mean = self._create_weight_copy()
        self._swag_deviations.clear()
        self._posterior_samples = None
        batches_between_updates = max(len(loader) // self.swag_updates_per_epoch, 1)

        self.network.train()
        with tqdm.trange(self.swag_training_epochs, desc="Running gradient descent for SWA") as pbar:
            progress_dict = {}
            for epoch in pbar:
                avg_loss = 0.0
                avg_accuracy = 0.0
                num_samples = 0
                for batch_index, (batch_images, batch_snow_labels, batch_cloud_labels, batch_labels) in enumerate(loader):
                    batch_targets = torch.nn.functional.one_hot(batch_labels, num_classes=6).float()
                    if self.swag_augment:
                        batch_images = random_dihedral_transform(batch_images)
                    if self.swag_cutmix_prob > 0 and torch.rand(()).item() < self.swag_cutmix_prob:
                        batch_images, batch_targets = cutmix(batch_images, batch_targets)

                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_targets)
                    batch_loss.backward()
                    optimizer.step()
                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (
                        num_samples + batch_images.size(0)
                    )
                    avg_accuracy = (
                        torch.sum(predictions.argmax(dim=-1) == batch_labels).item()
                        + num_samples * avg_accuracy
                    ) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)
                    progress_dict["avg. epoch loss"] = avg_loss
                    progress_dict["avg. epoch accuracy"] = avg_accuracy
                    pbar.set_postfix(progress_dict)

                    # After burn-in, collect `swag_updates_per_epoch` evenly spaced snapshots
                    # in every `swag_update_interval`-th epoch (the last one at the end of the epoch)
                    num_batches_done = batch_index + 1
                    if (
                        epoch >= self.swag_burn_in_epochs
                        and (epoch + 1) % self.swag_update_interval == 0
                        and num_batches_done % batches_between_updates == 0
                        and num_batches_done // batches_between_updates <= self.swag_updates_per_epoch
                    ):
                        self.update_swag_statistics()

    def calibrate_confidence(self, validation_data: torch.utils.data.Dataset) -> None:
        """
        Calibrate your predictions using a small validation set.
        validation_data contains well-defined and ambiguous samples,
        where you can identify the latter by having label -1.
        """
        if self.inference_mode == InferenceMode.MAP:
            # In MAP mode, simply predict argmax and do nothing else
            self._calibration_threshold = 0.0
            return

        # Predicting a label with confidence p costs 3 * (1 - p) in expectation, "don't know" always costs 1.
        # Hence, for calibrated probabilities, predicting is optimal iff p >= 2/3.
        self._calibration_threshold = 2.0 / 3.0

        val_images, val_snow_labels, val_cloud_labels, val_labels = validation_data.tensors
        assert val_images.size() == (140, 3, 60, 60)  # N x C x H x W
        assert val_labels.size() == (140,)
        assert val_snow_labels.size() == (140,)
        assert val_cloud_labels.size() == (140,)

        # Temperature scaling: a single scalar T divides every sampled network's logits before the softmax.
        # Ambiguous samples (label -1) have no target class, so plain NLL cannot use them.
        # Instead, we minimize NLL on well-defined samples plus the cross-entropy between the uniform distribution
        # and the prediction on ambiguous samples, which explicitly penalizes confidence on ambiguous images.
        # We only allow T >= 1 (never sharpen): the test set contains more ambiguity than the validation set,
        # and an ECE-based fit that sharpened the predictions (T = 0.5) failed the public test set.
        self.network.eval()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_images),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        with torch.no_grad():
            val_logits = torch.stack(self._predict_logits_per_sample(loader))  # S x V x N x C

        ambiguous_mask = val_labels == -1
        best_loss = None
        for temperature in np.round(np.arange(1.0, 5.01, 0.05), 2):
            probabilities = torch.softmax(val_logits / float(temperature), dim=-1).mean(dim=(0, 1))
            log_probabilities = torch.log(torch.clamp(probabilities, min=1e-12))
            nll = -log_probabilities[~ambiguous_mask, val_labels[~ambiguous_mask]].sum()
            uniform_cross_entropy = -log_probabilities[ambiguous_mask].mean(dim=-1).sum()
            loss = (nll + uniform_cross_entropy).item() / val_labels.size(0)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                self._temperature = float(temperature)
        print(f"Calibrated temperature {self._temperature:.2f} with validation loss {best_loss:.4f}")

    def predict_probabilities_swag(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Perform Bayesian model averaging using your SWAG statistics and predict
        probabilities for all samples in the loader.
        Outputs should be a Nx6 tensor, where N is the number of samples in loader,
        and all rows of the output should sum to 1.
        That is, output row i column j should be your predicted p(y=j | x_i).
        """

        self.network.eval()

        # Perform Bayesian model averaging:
        # Instead of sampling self.num_bma_samples networks (using self.sample_parameters())
        # for each datapoint, you can save time by sampling self.num_bma_samples networks,
        # and perform inference with each network on all samples in loader.
        # Each posterior sample's prediction averages over its test-time augmentation views
        model_predictions = [
            torch.softmax(sample_logits / self._temperature, dim=-1).mean(dim=0)
            for sample_logits in self._predict_logits_per_sample(loader)
        ]

        assert len(model_predictions) == self.num_bma_samples
        assert all(
            isinstance(sample_predictions, torch.Tensor)
            and sample_predictions.dim() == 2  # N x C
            and sample_predictions.size(1) == 6
            for sample_predictions in model_predictions
        )

        # BMA: average the predictive distributions (not the logits) of all posterior samples
        bma_probabilities = torch.stack(model_predictions).mean(dim=0)

        assert bma_probabilities.dim() == 2 and bma_probabilities.size(1) == 6  # N x C
        return bma_probabilities

    def _draw_posterior_samples(self) -> None:
        """
        Draw self.num_bma_samples networks from the SWAG posterior once, and store their state dicts
        (including the refitted batchnorm statistics). Validation calibration and test prediction then
        use the exact same posterior samples, and batchnorm statistics are only recomputed once per sample.
        """
        self._posterior_samples = []
        for _ in tqdm.trange(self.num_bma_samples, desc="Sampling networks from the SWAG posterior"):
            self.sample_parameters()
            self._posterior_samples.append(copy.deepcopy(self.network.state_dict()))

    def _predict_logits_per_sample(self, loader: torch.utils.data.DataLoader) -> typing.List[torch.Tensor]:
        """
        Return the (untempered) V x N x C logits of every cached posterior sample for all samples in loader,
        where V is the number of test-time augmentation views that sample uses.
        """
        if self._posterior_samples is None:
            self._draw_posterior_samples()

        all_logits = []
        for sample_index, state_dict in enumerate(self._posterior_samples):
            if self.test_time_augmentation == "full":
                views = range(NUM_DIHEDRAL_TRANSFORMS)
            elif self.test_time_augmentation == "cycle":
                views = [sample_index % NUM_DIHEDRAL_TRANSFORMS]
            else:
                views = [0]
            self.network.load_state_dict(state_dict)
            self.network.eval()
            all_logits.append(torch.stack([
                torch.cat([self.network(dihedral_transform(batch_images, view)) for (batch_images,) in loader])
                for view in views
            ]))
        return all_logits

    def sample_parameters(self) -> None:
        """
        Sample a new network from the approximate SWAG posterior.
        For simplicity, this method directly modifies self.network in-place.
        Hence, after calling this method, self.network corresponds to a new posterior sample.
        """

        use_low_rank = self.inference_mode == InferenceMode.SWAG_FULL and len(self._swag_deviations) > 1
        if use_low_rank:
            # The low-rank term acts on the full weight vector,
            # so a single z_low_rank in R^K must be shared by all layers.
            rank = len(self._swag_deviations)
            z_low_rank = torch.randn(rank)

        # Instead of acting on a full vector of parameters, all operations can be done on per-layer parameters.
        for name, param in self.network.named_parameters():
            # SWAG-diagonal part
            z_diag = torch.randn(param.size())
            mean_weights = self._swag_mean[name]
            # Clamp: E[theta^2] - E[theta]^2 can be slightly negative due to floating point errors
            variance = torch.clamp(self._swag_sq_mean[name] - mean_weights ** 2, min=1e-30)
            std_weights = torch.sqrt(variance)
            if use_low_rank:
                # Full SWAG covariance is (Sigma_diag + D D^T / (K - 1)) / 2
                std_weights = std_weights / math.sqrt(2.0)
            assert mean_weights.size() == param.size() and std_weights.size() == param.size()

            # Diagonal part
            sampled_weight = mean_weights + std_weights * z_diag

            # Full SWAG part
            if use_low_rank:
                # Per-layer slice of the deviation matrix D, shape (*param.size(), K)
                deviation_matrix = torch.stack([deviation[name] for deviation in self._swag_deviations], dim=-1)
                sampled_weight += (deviation_matrix @ z_low_rank) / math.sqrt(2.0 * (rank - 1))

            # Modify weight value in-place; directly changing self.network
            param.data = sampled_weight

        # Batchnorm statistics of the sampled network differ from those stored during training
        self._update_batchnorm_statistics()

    def decide_labels(self, predicted_probabilities: torch.Tensor) -> torch.Tensor:
        """
        Predict labels in {0, 1, 2, 3, 4, 5} or "don't know" as -1
        based on your model's predicted probabilities.
        The parameter predicted_probabilities is an Nx6 tensor containing predicted probabilities
        as returned by predict_probabilities(...).
        The output should be a N-dimensional long tensor, containing values in {-1, 0, 1, 2, 3, 4, 5}.
        """

        # label_probabilities contains the per-row maximum values in predicted_probabilities,
        # max_likelihood_labels the corresponding column index (equivalent to class).
        label_probabilities, max_likelihood_labels = torch.max(predicted_probabilities, dim=-1)
        num_samples, num_classes = predicted_probabilities.size()
        assert label_probabilities.size() == (num_samples,) and max_likelihood_labels.size() == (num_samples,)

        # A model without uncertainty awareness might simply predict the most likely label per sample:
        # return max_likelihood_labels

        # A bit better: use a threshold to decide whether to return a label or "don't know" (label -1)
        # TODO(2): implement a different decision rule if desired
        return torch.where(
            label_probabilities >= self._calibration_threshold,
            max_likelihood_labels,
            torch.ones_like(max_likelihood_labels) * -1,
        )

    def _create_weight_copy(self) -> typing.Dict[str, torch.Tensor]:
        """Create an all-zero copy of the network weights as a dictionary that maps name -> weight"""
        return {
            name: torch.zeros_like(param, requires_grad=False)
            for name, param in self.network.named_parameters()
        }

    def run_training(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> None:
        """
        Perform full SWAG fitting procedure.
        If `USE_PRETRAINED_WEIGHTS` is `True`, this method skips the MAP inference part,
        and uses pretrained weights instead.

        Note that MAP inference can take a very long time.
        You should hence only perform MAP inference yourself after passing the baseline
        using the given CNN architecture and pretrained weights.
        """

        # MAP inference to obtain initial weights
        PRETRAINED_WEIGHTS_FILE = self.weights_dir / "map_weights.pt"
        if USE_PRETRAINED_WEIGHTS:
            self.network.load_state_dict(torch.load(PRETRAINED_WEIGHTS_FILE))
            print("Loaded pretrained MAP weights from", PRETRAINED_WEIGHTS_FILE)
        else:
            self.fit_map_model(loader)

        # SWAG
        if self.inference_mode in (InferenceMode.SWAG_DIAGONAL, InferenceMode.SWAG_FULL):
            self.fit_swag_model(loader)

    def fit_map_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        MAP inference procedure to obtain initial weights of self.network.
        This is the exact procedure that was used to obtain the pretrained weights we provide.
        """
        map_training_epochs = 140
        initial_learning_rate = 0.01
        reduced_learning_rate = 0.0001
        start_decay_epoch = 50
        decay_factor = reduced_learning_rate / initial_learning_rate

        # Create optimizer, loss, and a learning rate scheduler that aids convergence
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=initial_learning_rate,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=decay_factor,
                    total_iters=(map_training_epochs - start_decay_epoch) * len(loader),
                ),
            ],
            milestones=[start_decay_epoch * len(loader)],
        )

        # Put network into training mode
        # Batch normalization layers are only updated if the network is in training mode,
        # and are replaced by a moving average if the network is in evaluation mode.
        self.network.train()
        with tqdm.trange(map_training_epochs, desc="Fitting initial MAP weights") as pbar:
            progress_dict = {}
            # Perform the specified number of MAP epochs
            for epoch in pbar:
                avg_loss = 0.0
                avg_accuracy = 0.0
                num_samples = 0
                # Iterate over batches of randomly shuffled training data
                for batch_images, _, _, batch_labels in loader:
                    # Training step
                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_labels)
                    batch_loss.backward()
                    optimizer.step()

                    # Save learning rate that was used for step, and calculate new one
                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    with warnings.catch_warnings():
                        # Suppress annoying warning (that we cannot control) inside PyTorch
                        warnings.simplefilter("ignore")
                        lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (
                        num_samples + batch_images.size(0)
                    )
                    avg_accuracy = (
                        torch.sum(predictions.argmax(dim=-1) == batch_labels).item()
                        + num_samples * avg_accuracy
                    ) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)

                    progress_dict["avg. epoch loss"] = avg_loss
                    progress_dict["avg. epoch accuracy"] = avg_accuracy
                    pbar.set_postfix(progress_dict)

    def predict_probabilities(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for the given images xs.
        This method returns an NxC float tensor,
        where row i column j corresponds to the probability that y_i is class j.

        This method uses different strategies depending on self.inference_mode.
        """
        self.network = self.network.eval()

        # Create a loader that we can deterministically iterate many times if necessary
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        with torch.no_grad():  # save memory by not tracking gradients
            if self.inference_mode == InferenceMode.MAP:
                return self.predict_probabilities_map(loader)
            else:
                return self.predict_probabilities_swag(loader)

    def predict_probabilities_map(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Predict probabilities assuming that self.network is a MAP estimate.
        This simply performs a forward pass for every batch in `loader`,
        concatenates all results, and applies a row-wise softmax.
        """
        all_predictions = []
        for (batch_images,) in loader:
            all_predictions.append(self.network(batch_images))

        all_predictions = torch.cat(all_predictions)
        return torch.softmax(all_predictions, dim=-1)

    def _update_batchnorm_statistics(self) -> None:
        """
        Reset and fit batch normalization statistics using the training dataset self.training_dataset.
        We provide this method for you for convenience.
        See the SWAG paper for why this is required.

        Batch normalization usually uses an exponential moving average, controlled by the `momentum` parameter.
        However, we are not training but want the statistics for the full training dataset.
        Hence, setting `momentum` to `None` tracks a cumulative average instead.
        The following code stores original `momentum` values, sets all to `None`,
        and restores the previous hyperparameters after updating batchnorm statistics.
        """

        original_momentum_values = dict()
        for module in self.network.modules():
            # Only need to handle batchnorm modules
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue

            # Store old momentum value before removing it
            original_momentum_values[module] = module.momentum
            module.momentum = None

            # Reset batch normalization statistics
            module.reset_running_stats()

        loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        self.network.train()
        for (batch_images,) in loader:
            self.network(batch_images)
        self.network.eval()

        # Restore old `momentum` hyperparameter values
        for module, momentum in original_momentum_values.items():
            module.momentum = momentum


NUM_DIHEDRAL_TRANSFORMS = 8


def dihedral_transform(images: torch.Tensor, index: int) -> torch.Tensor:
    """
    Apply element `index` in {0, ..., 7} of the dihedral group D4 to a batch of N x C x H x W images:
    a rotation by (index mod 4) * 90 degrees, followed by a horizontal flip if index >= 4. Index 0 is the identity.
    Satellite images have no canonical orientation, so all 8 views show the same scene.
    """
    images = torch.rot90(images, index % 4, dims=(2, 3))
    if index >= 4:
        images = torch.flip(images, dims=(3,))
    return images


def random_dihedral_transform(images: torch.Tensor) -> torch.Tensor:
    """Apply an independent, uniformly random dihedral transform to every image in the batch."""
    transform_indices = torch.randint(NUM_DIHEDRAL_TRANSFORMS, (images.size(0),))
    transformed = images.clone()
    for index in range(NUM_DIHEDRAL_TRANSFORMS):
        mask = transform_indices == index
        if mask.any():
            transformed[mask] = dihedral_transform(images[mask], index)
    return transformed


def cutmix(images: torch.Tensor, targets: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    CutMix (Yun et al., 2019): paste a random box from a shuffled copy of the batch into every image,
    and mix the (soft) targets proportionally to the pasted area.
    Mixed images resemble the ambiguous test images that show multiple land usage types.
    """
    _, _, height, width = images.size()
    permutation = torch.randperm(images.size(0))
    mixing_ratio = torch.distributions.Beta(1.0, 1.0).sample().item()
    box_height = int(height * math.sqrt(1.0 - mixing_ratio))
    box_width = int(width * math.sqrt(1.0 - mixing_ratio))
    center_y = torch.randint(height, ()).item()
    center_x = torch.randint(width, ()).item()
    top, bottom = max(center_y - box_height // 2, 0), min(center_y + box_height // 2, height)
    left, right = max(center_x - box_width // 2, 0), min(center_x + box_width // 2, width)

    mixed_images = images.clone()
    mixed_images[:, :, top:bottom, left:right] = images[permutation, :, top:bottom, left:right]
    pasted_fraction = (bottom - top) * (right - left) / (height * width)
    mixed_targets = (1.0 - pasted_fraction) * targets + pasted_fraction * targets[permutation]
    return mixed_images, mixed_targets


class SWAGScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Custom learning rate scheduler that calculates a different learning rate each gradient descent step.
    The default implementation keeps the original learning rate constant, i.e., does nothing.
    You can implement a custom schedule inside calculate_lr,
    and add+store additional attributes in __init__.
    You should not change any other parts of this class.
    """

    def calculate_lr(self, current_epoch: float, previous_lr: float) -> float:
        """
        Calculate the learning rate for the epoch given by current_epoch.
        current_epoch is the fractional epoch of SWA fitting, starting at 0.
        That is, an integer value x indicates the start of epoch (x+1),
        and non-integer values x.y correspond to steps in between epochs (x+1) and (x+2).
        previous_lr is the previous learning rate.

        This method should return a single float: the new learning rate.
        """
        # TODO(2): Implement a custom schedule if desired
        return previous_lr

    # TODO(2): Add and store additional arguments if you decide to implement a custom scheduler
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        steps_per_epoch: int,
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        super().__init__(optimizer, last_epoch=-1, verbose=False)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )
        return [
            self.calculate_lr(self.last_epoch / self.steps_per_epoch, group["lr"])
            for group in self.optimizer.param_groups
        ]


def evaluate_model(
    swag_inference: SWAGPredictor,
    eval_dataset: torch.utils.data.Dataset,
    extended_evaluation: bool,
    output_location: pathlib.Path,
) -> None:
    """
    Run evaluation with your model.
    Feel free to change or extend this code.
    :param swag_inference: Trained model to evaluate
    :param eval_dataset: Validation dataset
    :param: extended_evaluation: If True, generates additional plots
    :param output_location: Directory into which extended evaluation plots are saved
    """

    print("Evaluating model on validation data")

    # We ignore is_snow and is_cloud here, but feel free to use them as well
    images, snow_labels, cloud_labels, labels = eval_dataset.tensors

    # Predict class probabilities on test data,
    # most likely classes (according to the max predicted probability),
    # and classes as predicted by your SWAG implementation.
    all_pred_probabilities = swag_inference.predict_probabilities(images)
    max_pred_probabilities, argmax_pred_labels = torch.max(all_pred_probabilities, dim=-1)
    predicted_labels = swag_inference.decide_labels(all_pred_probabilities)

    # Create a mask that ignores ambiguous samples (those with class -1)
    non_ambiguous_mask = labels != -1

    # Calculate three kinds of accuracy:
    # 1. Overall accuracy, counting "don't know" (-1) as its own class
    # 2. Accuracy on all samples that have a known label. Predicting -1 on those counts as wrong here.
    # 3. Accuracy on all samples that have a known label w.r.t. the class with the highest predicted probability.
    overall_accuracy = torch.mean((predicted_labels == labels).float()).item()
    non_ambiguous_accuracy = torch.mean((predicted_labels[non_ambiguous_mask] == labels[non_ambiguous_mask]).float()).item()
    non_ambiguous_argmax_accuracy = torch.mean(
        (argmax_pred_labels[non_ambiguous_mask] == labels[non_ambiguous_mask]).float()
    ).item()
    print(f"Accuracy (raw): {overall_accuracy:.4f}")
    print(f"Accuracy (non-ambiguous only, your predictions): {non_ambiguous_accuracy:.4f}")
    print(f"Accuracy (non-ambiguous only, predicting most-likely class): {non_ambiguous_argmax_accuracy:.4f}")

    # Determine which threshold would yield the smallest cost on the validation data
    # Note that this threshold does not necessarily generalize to the test set!
    # However, it can help you judge your method's calibration.
    threshold_values = [0.0] + list(torch.unique(max_pred_probabilities, sorted=True))
    costs = []
    for threshold in threshold_values:
        thresholded_predictions = torch.where(max_pred_probabilities <= threshold, -1 * torch.ones_like(predicted_labels), predicted_labels)
        costs.append(compute_cost(thresholded_predictions, labels).item())
    best_threshold_index = np.argmin(costs)
    print(f"Best cost {costs[best_threshold_index]} at threshold {threshold_values[best_threshold_index]}")
    print("Note that this threshold does not necessarily generalize to the test set!")

    # Calculate ECE and plot the calibration curve
    calibration_data = calculate_calibration_curve(all_pred_probabilities.numpy(), labels.numpy(), num_bins=20)
    print("Validation ECE:", calibration_data["ece"])

    if extended_evaluation:
        print("Plotting reliability diagram")
        fig = paint_reliability_diagram(calibration_data)
        fig.savefig(output_location / "reliability_diagram.pdf")

        sorted_confidence_indices = torch.argsort(max_pred_probabilities)

        # Plot samples your model is most confident about
        print("Plotting most confident validation set predictions")
        most_confident_indices = sorted_confidence_indices[-10:]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_index = most_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f"pred. {predicted_labels[sample_index]}, true {labels[sample_index]}")
                bar_colors = ["C0"] * 6
                if labels[sample_index] >= 0:
                    bar_colors[labels[sample_index]] = "C1"
                ax[row + 1, col].bar(
                    np.arange(6), all_pred_probabilities[sample_index].numpy(), tick_label=np.arange(6), color=bar_colors
                )
        fig.suptitle("Most confident predictions", size=20)
        fig.savefig(output_location / "examples_most_confident.pdf")

        # Plot samples your model is least confident about
        print("Plotting least confident validation set predictions")
        least_confident_indices = sorted_confidence_indices[:10]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_index = least_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f"pred. {predicted_labels[sample_index]}, true {labels[sample_index]}")
                bar_colors = ["C0"] * 6
                if labels[sample_index] >= 0:
                    bar_colors[labels[sample_index]] = "C1"
                ax[row + 1, col].bar(
                    np.arange(6), all_pred_probabilities[sample_index].numpy(), tick_label=np.arange(6), color=bar_colors
                )
        fig.suptitle("Least confident predictions", size=20)
        fig.savefig(output_location / "examples_least_confident.pdf")


class CNN(torch.nn.Module):
    """
    Small convolutional neural network used in this task.
    You should not modify this class before passing the baseline.

    Note that if you change the architecture of this network,
    you need to re-run MAP inference and cannot use the provided pretrained weights anymore.
    Hence, you need to set `USE_PRETRAINED_WEIGHTS = False` at the top of this file.
    """
    def __init__(
        self,
        in_channels: int,
        out_classes: int,
    ):
        super().__init__()

        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=5),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.pool1 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.pool2 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
        )

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.linear = torch.nn.Linear(64, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = self.layer5(x)

        # Average features over both spatial dimensions, and remove the now superfluous dimensions
        x = self.global_pool(x).squeeze(-1).squeeze(-1)

        log_softmax = self.linear(x)

        return log_softmax


if __name__ == "__main__":
    main()
