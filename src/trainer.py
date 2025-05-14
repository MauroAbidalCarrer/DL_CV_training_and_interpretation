from typing import Callable
from datetime import datetime
from functools import reduce
from itertools import accumulate
from IPython.display import display
from dataclasses import dataclass, field

import torch
import pandas as pd
from numpy import mean
from torch.optim import SGD
from plotly.express import line
from pandas import DataFrame as DF
from torch.nn import Module, CrossEntropyLoss
from plotly.graph_objects import FigureWidget
from torch.utils.data import DataLoader as DL


metric_fn = Callable[[dict], dict]


@dataclass
class Trainer:
    model:Module
    optimizer:SGD
    epoch: int = field(default=0, init=False)
    step_count: int = field(default=0, init=False)
    loss:CrossEntropyLoss = field(default_factory=CrossEntropyLoss())
    training_metrics:list[dict] = field(default_factory=list, init=False)

    def optimize_nn(self, epochs, train_dl:DL, test_dl:DL, catch_key_int=True, **plt_kwargs) -> DF:
        """Optimizes the neural network and returns a dataframe of the training metrics."""
        if catch_key_int:
            try:
                self._optimize_nn(epochs, train_dl, test_dl, plt_kwargs)
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt exception, returning training metrics.")
        else:
            self._optimize_nn(epochs, train_dl, test_dl, plt_kwargs)
        return DF.from_records(self.training_metrics)

    def _optimize_nn(self, epochs, train_dl:DL, test_dl:DL, plt_kwargs=None):
        fig = None
        model_device = next(self.model.parameters()).device
        metrics:list[dict] = []
        # Use epoch instead of epoch.
        # This avoids resetting new metrics DF lines to the same epoch value in case this method gets recalled.
        for _ in range(epochs):
            metrics.append(self.record_metrics(self.model, train_dl, test_dl))
            if plt_kwargs is not None:
                fig = self.create_figure_widget(plt_kwargs) if fig is None else fig
                self.update_figure(fig, plt_kwargs)
            for batch_x, batch_y in train_dl:
                self.optimizer.zero_grad()
                batch_x = batch_x.to(model_device)
                outputs = self.model(batch_x)
                loss_val = self.loss(outputs, batch_y)
                loss_val.backward()
                self.optimizer.step(batch_x, batch_y.to(model_device))
            self.epoch += 1

    def record_metrics(self, train_dl:DL, test_dl:DL) -> dict[str, any]:
        with torch.no_grad():
            return {
                "epoch": self.epoch,
                "step": self.step_count,
                **self.metrics_of_dataset(test_dl, "test"),
                **self.metrics_of_dataset(train_dl, "train"),
                **self.optimizer.state_dict()["param_groups"],
            }

    def metrics_of_dataset(self, data_loader:DL, dl_prefix:str) -> dict:
        model_device = next(self.model.parameters()).device
        losses = []
        accuracies = []
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(model_device)
            batch_y = batch_y.to(model_device)
            losses.append(self.loss(self.model(batch_x), batch_y).item())            
            accuracies.append((torch.max(self.model(batch_x), 1)[1] == batch_y).sum().item())
        return {
            dl_prefix + "_loss": mean(losses),
            dl_prefix + "_accuracy":mean(accuracies)
        }
    
    def create_figure_widget(self, plt_kwargs:dict) -> FigureWidget:
        df = (
            DF.from_records(self.training_metrics)
            .melt(plt_kwargs["x"], plt_kwargs["y"])
        )
        fig = (
            line(
                df,
                plt_kwargs["x"],
                "value",
                facet_row="variable",
                color="variable",
                markers=True,
                **plt_kwargs
            )
            .update_yaxes(matches=None)
        )
        fig = FigureWidget(fig)
        display(fig)
        return fig

    def update_figure(self, fig:FigureWidget, plt_kwargs:dict):
        df = DF.from_records(self.training_metrics)
        with fig.batch_update():
            for i, plt_y in enumerate(plt_kwargs["y"]):
                fig.data[i].x = df[plt_kwargs["x"]]
                fig.data[i].y = df[plt_y]
