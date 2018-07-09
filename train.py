import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from monitor import batch_acc, timeSince
import config
import data
import model

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.epoch = 0
        self.iterations = 0
        self.accuracies = np.array([])
        self.losses = np.array([])
        self.eval_accuracies = np.array([])
        self.log_softmax = nn.LogSoftmax(dim=0).to(DEVICE)

    def update_lr(self):
        lr = config.initial_learning_rate * \
            0.5 ** (float(self.iterations) / config.lr_halflife)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def run_epoch(self, loader, print_every):
        self.epoch += 1
        self.model.train()

        start_time = time.time()
        for i, (img, q, a, _, q_len) in enumerate(loader):
            img = Variable(img.to(DEVICE), requires_grad=False)
            q = Variable(q.to(DEVICE), requires_grad=False)
            a = Variable(a.to(DEVICE), requires_grad=False)
            q_len = Variable(q_len.to(DEVICE), requires_grad=False)

            out = self.model(img, q, q_len)
            nll = -self.log_softmax(out)
            loss = (nll * a / 10).sum(dim=1).mean()
            acc = batch_acc(out.data, a.data).cpu()

            self.update_lr()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iterations += 1

            self.losses = np.append(self.losses, loss.item())
            self.accuracies = np.append(self.accuracies, acc.mean())

            if i % print_every == 0:
                print('Epoch %d %d%% (%s) avg_loss: %.4f avg_acc: %.4f' % (
                    self.epoch, i / len(loader), timeSince(start_time), np.mean(self.losses), np.mean(self.accuracies)))

    def eval(self, loader):
        self.model.eval()

        final_acc = 0.0
        start_time = time.time()
        for i, (img, q, a, _, q_len) in enumerate(loader):
            img = Variable(img.to(DEVICE), requires_grad=False)
            q = Variable(q.to(DEVICE), requires_grad=False)
            a = Variable(a.to(DEVICE), requires_grad=False)
            q_len = Variable(q_len.to(DEVICE), requires_grad=False)

            out = self.model(img, q, q_len)
            acc = batch_acc(out.data, a.data).cpu()

            final_acc += (acc.mean() * img.size(0)) / len(loader.dataset)

        self.eval_accuracies = np.append(self.eval_accuracies, final_acc)
        print('Epoch %d %s accuracy on eval set: %d%%' %
              (self.epoch, timeSince(start_time), final_acc))


def main():
    # train_loader = data.create_vqa_loader(train=True)
    eval_loader = data.create_vqa_loader(train=False)

    main_model = model.MainModel(eval_loader.dataset.num_tokens()).to(DEVICE)
    optimizer = optim.Adam(
        [p for p in main_model.parameters() if p.requires_grad])

    trainer = Trainer(main_model, optimizer)
    for i in range(config.epochs):
        # trainer.run_epoch(train_loader, print_every=1)
        trainer.eval(eval_loader)


if __name__ == '__main__':
    main()
