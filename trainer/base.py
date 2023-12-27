from abc import ABC
import os
from datetime import datetime
import torch
from torch.utils import tensorboard
from tqdm import tqdm

class BaseTrainer(ABC):
    def __init__(self, model,
                 train_loader, val_loader, test_loader,
                 loss_fn, metric_fn, optimizer, scheduler,
                 accelerator
                 ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
    
    def resume(self):
        if not os.path.exists(self.resume_path): return
        train_record = torch.load(self.resume_path)
        self.cur_epoch = train_record['epoch']
        self.writer_step = train_record['step']
        self.model.load_state_dict(train_record['model_state_dict'])
        self.optimizer.load_state_dict(train_record['optimizer_state_dict'])
    
    def save(self):
        if not self.train_record: return
        torch.save(self.train_record, self.resume_path)

    def train_one_epoch(self):
        result = {}
        self.train_loss = 0.0
        self.model.train()
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(self.accelerator), labels.to(self.accelerator)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            self.on_end_train_one_batch(batch_idx, loss)

        return result

    def on_end_train_one_batch(self, batch_idx, loss):
        self.train_loss += loss.item()
        if batch_idx % self.log_train_per_batchs == self.log_train_per_batchs-1:
            loss_val = self.train_loss / self.log_train_per_batchs
            self.writer.add_scalar('train_loss', loss_val)
            self.writer_step += 1
            self.progress_bar_postfix.update({
                'train_loss': f'{loss_val:.3f}'
            })
            self.progress_bar.set_postfix(self.progress_bar_postfix)
            self.train_loss = 0.0

    def test_one_epoch(self):
        result = {}
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for batch in self.test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.accelerator), labels.to(self.accelerator)
                outputs = self.model(inputs)
                correct += self.metric_fn(outputs, labels)
        correct /= len(self.test_loader.dataset)
        result.update({
            'correct': correct
        })
        self.writer.add_scalar('test_correct', correct)
        self.progress_bar_postfix.update({
            'test_correct': f'{correct:.3f}'
        })
        self.progress_bar.set_postfix(self.progress_bar_postfix)
        return result

    def run(self, epochs=1, begin_from=None, root_log_dir='./logs',
            log_train_per_batchs=1):
        self.log_train_per_batchs = log_train_per_batchs

        if begin_from == None:
            begin_from = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.log_dir = os.path.join(root_log_dir, begin_from)
        self.writer = tensorboard.SummaryWriter(log_dir=self.log_dir)
        self.writer_step = 0
        self.resume_path = os.path.join(self.log_dir, 'resume.pth')
        self.cur_epoch = 0
        self.resume()

        self.model.to(self.accelerator)

        self.progress_bar = tqdm(range(self.cur_epoch, epochs))
        self.progress_bar_postfix = {}
        for epoch in self.progress_bar:
            train_result_one_epoch = self.train_one_epoch()
            test_result_one_epoch = self.test_one_epoch()

            self.cur_epoch += 1
            self.train_record = {
                'epoch': self.cur_epoch,
                'step': self.writer_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }
            self.save()