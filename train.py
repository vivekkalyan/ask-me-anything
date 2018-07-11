import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt

from monitor import batch_acc, timeSince
import config
import data
import model

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.epoch = 0
        self.iterations = 0
        self.accuracies = np.array([])
        self.losses = np.array([])
        self.eval_accuracies = np.array([])
        self.log_softmax = nn.LogSoftmax(dim=1).to(DEVICE)

    def update_lr(self):
        lr = config.initial_learning_rate * \
            0.5 ** (float(self.iterations) / config.lr_halflife)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def run_epoch(self, loader, print_every, plot_every):
        self.epoch += 1
        self.model.train()

        start_time = time.time()
        print('------------***Epoch %d***------------' % self.epoch)
        for i, (img, q, a, _, q_len) in enumerate(loader):
            img = img.to(DEVICE)
            q = q.to(DEVICE)
            a = a.to(DEVICE)
            q_len = q_len.to(DEVICE)

            out = self.model(img, q, q_len)
            nll = -self.log_softmax(out)
            loss = (nll * a / 10).sum(dim=1).mean()
            acc = batch_acc(out.data, a.data).cpu()

            self.update_lr()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iterations += 1

            if i % plot_every == 0:
                self.losses = np.append(self.losses, loss.item())
                self.accuracies = np.append(self.accuracies, acc.mean())

            if i % print_every == 0:
                print('Epoch %d %d%% (%s) avg_loss: %.4f avg_acc: %.4f' % (
                    self.epoch, float(i) / len(loader) * 100, timeSince(start_time), np.mean(self.losses), np.mean(self.accuracies)))

    def eval(self, loader):
        self.model.eval()

        final_acc = 0.0
        start_time = time.time()
        print('Evaluating the model...')
        for img, q, a, _, q_len in tqdm(loader):
            img = img.to(DEVICE)
            q = q.to(DEVICE)
            a = a.to(DEVICE)
            q_len = q_len.to(DEVICE)

            out = self.model(img, q, q_len)
            acc = batch_acc(out.data, a.data).cpu()

            final_acc += float(acc.mean() * img.size(0)) / len(loader.dataset)

        self.eval_accuracies = np.append(self.eval_accuracies, final_acc)
        print('Epoch %d %s accuracy on eval set: %.2f%%' %
              (self.epoch, timeSince(start_time), final_acc * 100))

    def save_checkpoint(self, filename):
        torch.save({
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.losses,
            'accuracies': self.accuracies,
            'eval_accuracies': self.eval_accuracies,
        }, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.losses = checkpoint['losses']
        self.accuracies = checkpoint['accuracies']
        self.eval_accuracies = checkpoint['eval_accuracies']


def main():
    train_loader = data.create_vqa_loader(train=True)
    eval_loader = data.create_vqa_loader(train=False)

    main_model = model.MainModel(train_loader.dataset.num_tokens())
    optimizer = optim.Adam(
        [p for p in main_model.parameters() if p.requires_grad])

    trainer = Trainer(main_model, optimizer)
    plot_every = 100
    for i in range(config.epochs):
        trainer.run_epoch(train_loader, print_every=200, plot_every=plot_every)
        trainer.eval(eval_loader)
        trainer.save_checkpoint(
            'VQA_with_Attention_epoch_' + str(trainer.epoch))

    plt.plot(np.array(range(len(trainer.losses))) * plot_every, trainer.losses)
    plt.title('Training losses of Model without Attention')
    plt.xlab('Iterations')
    plt.ylab('Losses')
    plt.show()

    plt.plot(trainer.eval_accuracies)
    plt.title('Eval acccuracies of Model without Attention')
    plt.xlab('Epochs')
    plt.ylab('Accuracies')
    plt.show()


if __name__ == '__main__':
    main()
