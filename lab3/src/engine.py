import csv
from datetime import datetime

import numpy as np
from tqdm import tqdm


class Engine:
    def __init__(self, model, device, logger_path, loss_func, optimizer, lr_scheduler, metric):

        self.model = model
        self.logger_path = logger_path
        with open(self.logger_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "val_loss", f"train_{metric.printout_name}", f"val_{metric.printout_name}"])
            writer.writeheader()


        self.loss = loss_func

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metric = metric

        self.device=device
        self.model.to(device)


    def log(self, epoch, train_loss, val_loss, train_acc, val_acc):
        with open(self.logger_path, 'a') as file:
            writer = csv.writer(file, delimiter=',')
            row = [epoch, train_loss, val_loss, train_acc, val_acc]
            writer.writerow(row)


    def train_ep(self, loader, epoch_number):
        self.model.train()

        t1 = datetime.now()
        loss_list = []

        prog_bar = tqdm(loader, total=len(loader))
        for x, y in prog_bar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            y_pred = self.model(x)
            loss = self.loss(y_pred, y)
            loss_list += [loss.cpu().detach().numpy()]

            loss.backward()
            self.optimizer.step()

            self.metric.update_per_batch(y_pred.detach().cpu().numpy(), y.cpu().numpy())

            prog_bar.set_description(f"Loss: {loss:.5f}")

        self.lr_scheduler.step()
        t2 = datetime.now()
        metric_value = self.metric.calculate()
        self.metric.reset()

        print(f"TRAIN | Epoch {epoch_number}: mean loss={np.mean(loss_list):.3f}, " + 
              f"{self.metric.printout_name}={metric_value:.3f}, time: {(t2-t1).microseconds/1000:.3f} ms")
        return np.mean(loss_list), metric_value
    

    def validate_ep(self, loader, epoch_number):
        self.model.eval()

        t1 = datetime.now()
        loss_list = []

        prog_bar = tqdm(loader, total=len(loader))
        for x, y in prog_bar:
            x, y = x.to(self.device), y.to(self.device)

            y_pred = self.model(x)
            loss = self.loss(y_pred, y)
            loss_list += [loss.cpu().detach().numpy()]

            self.metric.update_per_batch(y_pred.cpu().detach().numpy(), y.cpu().numpy())

            prog_bar.set_description(f"Loss: {loss:.5f}")

        t2 = datetime.now()
        
        metric_value = self.metric.calculate()
        self.metric.reset()

        print(f"VAL | Epoch {epoch_number}: mean loss={np.mean(loss_list):.3f}, " + 
              f"{self.metric.printout_name}={metric_value:.3f}, time: {(t2-t1).microseconds/1000:.3f} ms")
        return np.mean(loss_list), metric_value

    
    def run(self, train_loader, val_loader, epochs):
        for e in range(epochs):
            print(f"Epoch {e}")
            train_loss, train_metric = self.train_ep(train_loader, epoch_number=e)
            val_loss, val_metric = self.validate_ep(val_loader, epoch_number=e)

            self.log(e, train_loss, val_loss, train_metric, val_metric)


