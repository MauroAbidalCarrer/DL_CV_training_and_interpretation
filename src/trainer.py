from datetime import datetime
from IPython.display import display
from dataclasses import dataclass, field

import torch
from torch.optim import Optimizer
from plotly.express import line
from pandas import DataFrame as DF
from torch.nn import Module, CrossEntropyLoss
from plotly.graph_objects import FigureWidget
from torch.utils.data import DataLoader as DL


@dataclass
class Trainer:
    model:Module
    optimizer:Optimizer
    fig: FigureWidget = field(default=None, init=False)
    step: int = field(default=0, init=False)
    epoch: int = field(default=0, init=False)
    loss:CrossEntropyLoss = field(default_factory=CrossEntropyLoss)
    training_metrics:list[dict] = field(default_factory=list, init=False)

    def optimize_nn(self, epochs, train_dl:DL, test_dl:DL, *, catch_key_int=True, plt_kwargs: dict=None) -> DF:
        """Optimizes the neural network and returns a dataframe of the training metrics."""
        if catch_key_int:
            try:
                self._training_loop(epochs, train_dl, test_dl, plt_kwargs)
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt exception, returning training metrics.")
        else:
            self._training_loop(epochs, train_dl, test_dl, plt_kwargs)
        return DF.from_records(self.training_metrics)

    def _training_loop(self, epochs, train_dl:DL, test_dl:DL, plt_kwargs=None):
        model_device = next(self.model.parameters()).device
        if model_device.type != "cuda":
            print("Warning: Model is not on a cuda device.")
        # Use self.epoch instead of for epoch in range(epochs).
        # This avoids resetting new metrics DF lines to the same epoch value in case this method gets recalled.
        if self.epoch == 0:
            self.record_and_display_metrics(train_dl, test_dl, plt_kwargs)
        for _ in range(epochs):
            self.model.train()
            total_loss = 0
            total_accuracy = 0
            for batch_x, batch_y in train_dl:
                self.optimizer.zero_grad()
                batch_x = batch_x.to(model_device, non_blocking=True)
                batch_y = batch_y.to(model_device, non_blocking=True)
                batch_y_pred = self.model(batch_x)
                loss_value = self.loss(batch_y_pred, batch_y)
                loss_value.backward()
                total_loss += loss_value.item()
                total_accuracy += (torch.max(batch_y_pred, 1)[1] == batch_y).sum().item()
                self.optimizer.step()
                self.step += 1
            self.epoch += 1
            train_metrics = {
                "train_loss": total_loss / len(train_dl),
                "train_accuracy": total_accuracy / len(train_dl.dataset)
            }
            self.record_and_display_metrics(train_metrics, test_dl, plt_kwargs)

    def record_and_display_metrics(self, train_dl: DL, test_dl: DL, plt_kwargs: dict):
        self.training_metrics.append(self.record_metrics(train_dl, test_dl))
        if plt_kwargs is not None:
            if self.fig is None:
                self.create_figure_widget(plt_kwargs)
            self.update_figure(plt_kwargs)

    def record_metrics(self, train_dl: DL|dict, test_dl: DL) -> dict[str, any]:
        # This is ugly but it does the trick
        if isinstance(train_dl, DL):
            train_metrics = self.metrics_of_dataset(train_dl, "train")
        else: 
            train_metrics = train_dl
        return {
            "epoch": self.epoch,
            "step": self.step,
            "date": datetime.now(),
            **self.metrics_of_dataset(test_dl, "test"),
            **train_metrics,
            **self.optimizer.state_dict()["param_groups"][-1],
        }

    @torch.no_grad()
    def metrics_of_dataset(self, data_loader: DL, dl_prefix: str) -> dict:
        self.model.eval()
        model_device = next(self.model.parameters()).device
        total_loss = 0
        total_accuracy = 0
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(model_device, non_blocking=True)
            batch_y = batch_y.to(model_device, non_blocking=True)
            batch_y_pred = self.model(batch_x)
            total_loss += self.loss(batch_y_pred, batch_y).item()
            total_accuracy += (torch.max(batch_y_pred, 1)[1] == batch_y).sum().item()
        return {
            # Divide loss by nb batches
            dl_prefix + "_loss": total_loss / len(data_loader),
            # Divide accuracy by nb elements
            dl_prefix + "_accuracy": total_accuracy / len(data_loader.dataset),
        }

    def create_figure_widget(self, plt_kwargs: dict) -> FigureWidget:
        df = (
            DF.from_records(self.training_metrics)
            .melt(plt_kwargs["x"], plt_kwargs["y"])
        )
        self.fig = (
            line(
                data_frame=df,
                y="value",
                facet_row="variable",
                color="variable",
                markers=True,
                **{k: v for k, v in plt_kwargs.items() if k != "y"},
            )
            .update_yaxes(matches=None)
        )
        self.fig = FigureWidget(self.fig)
        display(self.fig)

    def update_figure(self, plt_kwargs: dict):
        df = DF.from_records(self.training_metrics)
        with self.fig.batch_update():
            for i, plt_y in enumerate(plt_kwargs["y"]):
                self.fig.data[i].x = df[plt_kwargs["x"]]
                self.fig.data[i].y = df[plt_y]
