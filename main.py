from __future__ import print_function
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import numpy as np

def get_uniform_keys(n_keys, dim, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    keys = rng.uniform(-bound, bound, (n_keys, dim))
    return keys.astype(np.float32)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class HashingMemory(nn.Module):

    def __init__(self, input_dim, output_dim, params):

        super().__init__()

        # global parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_dim = params.k_dim
        self.v_dim = output_dim
        self.n_keys = params.n_keys
        self.size = self.n_keys ** 2
        self.heads = params.heads
        self.knn = params.knn
        assert self.k_dim >= 2 and self.k_dim % 2 == 0

        # dropout
        self.input_dropout = params.input_dropout
        self.query_dropout = params.query_dropout
        self.value_dropout = params.value_dropout

        # initialize keys / values
        self.initialize_keys()
        self.values = nn.EmbeddingBag(self.size, self.v_dim, mode='sum', sparse=params.sparse)
        nn.init.normal_(self.values.weight, mean=0, std=self.v_dim ** -0.5)

        # query network
        self.query_proj = nn.Sequential(*filter(None, [
            nn.Linear(self.input_dim, self.heads * self.k_dim, bias=True),
            nn.BatchNorm1d(self.heads * self.k_dim) if params.query_batchnorm else None
        ]))

        if params.query_batchnorm:
            print("WARNING: Applying batch normalization to queries improves the performance "
                  "and memory usage. But if you use it, be sure that you use batches of "
                  "sentences with the same size at training time (i.e. without padding). "
                  "Otherwise, the padding token will result in incorrect mean/variance "
                  "estimations in the BatchNorm layer.\n")

    def initialize_keys(self):
        """
        Create two subkey sets per head.
        `self.keys` is of shape (heads, 2, n_keys, k_dim // 2)
        """
        half = self.k_dim // 2
        keys = nn.Parameter(torch.from_numpy(np.array([
            get_uniform_keys(self.n_keys, half, seed=(2 * i + j))
            for i in range(self.heads)
            for j in range(2)
        ])).view(self.heads, 2, self.n_keys, half))
        self.keys = nn.Parameter(keys)

    def _get_indices(self, query, subkeys):
        """
        Generate scores and indices for a specific head.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        bs = query.size(0)
        knn = self.knn
        half = self.k_dim // 2
        n_keys = len(subkeys[0])

        # split query for product quantization
        q1 = query[:, :half]                                          # (bs,half)
        q2 = query[:, half:]                                          # (bs,half)

        # compute indices with associated scores
        scores1 = F.linear(q1, subkeys[0], bias=None)                 # (bs,n_keys)
        scores2 = F.linear(q2, subkeys[1], bias=None)                 # (bs,n_keys)
        scores1, indices1 = scores1.topk(knn, dim=1)                  # (bs,knn)
        scores2, indices2 = scores2.topk(knn, dim=1)                  # (bs,knn)

        # cartesian product on best candidate keys
        all_scores = (
            scores1.view(bs, knn, 1).expand(bs, knn, knn) +
            scores2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                # (bs,knn**2)
        all_indices = (
            indices1.view(bs, knn, 1).expand(bs, knn, knn) * n_keys +
            indices2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                # (bs,knn**2)

        # select best scores with associated indices
        scores, best_indices = torch.topk(all_scores, k=knn, dim=1)   # (bs,knn)
        indices = all_indices.gather(1, best_indices)                 # (bs,knn)

        assert scores.shape == indices.shape == (bs, knn)
        return scores, indices

    def get_indices(self, query):
        """
        Generate scores and indices.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        query = query.view(-1, self.heads, self.k_dim)
        bs = len(query)
        outputs = [self._get_indices(query[:, i], self.keys[i]) for i in range(self.heads)]
        s = torch.cat([s.view(bs, 1, self.knn) for s, _ in outputs], 1)  # (bs,heads,knn)
        i = torch.cat([i.view(bs, 1, self.knn) for _, i in outputs], 1)  # (bs,heads,knn)
        return s.view(-1, self.knn), i.view(-1, self.knn)

    def forward(self, input):
        """
        Read from the memory.
        """
        # input dimensions
        assert input.shape[-1] == self.input_dim
        prefix_shape = input.shape[:-1]
        bs = np.prod(prefix_shape)

        # compute query
        input = F.dropout(input, p=self.input_dropout, training=self.training)  # (...,i_dim)
        query = self.query_proj(input.contiguous().view(-1, self.input_dim))    # (bs,heads*k_dim)
        query = query.view(bs * self.heads, self.k_dim)                         # (bs*heads,k_dim)
        query = F.dropout(query, p=self.query_dropout, training=self.training)  # (bs*heads,k_dim)
        assert query.shape == (bs * self.heads, self.k_dim)

        # retrieve indices and scores
        scores, indices = self.get_indices(query)                               # (bs*heads,knn)
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)              # (bs*heads,knn)

        # merge heads / knn (since we sum heads)
        indices = indices.view(bs, self.heads * self.knn)                       # (bs,heads*knn)
        scores = scores.view(bs, self.heads * self.knn)                         # (bs,heads*knn)

        # weighted sum of values
        output = self.values(indices, per_sample_weights=scores)                # (bs,v_dim)
        output = F.dropout(output, p=self.value_dropout, training=self.training)# (bs,v_dim)

        # reshape output
        if len(prefix_shape) >= 2:
            output = output.view(prefix_shape + (self.v_dim,))                  # (...,v_dim)

        return output

    @staticmethod
    def register_args(parser):
        """
        Register memory parameters.
        """
        # memory parameters
        parser.add_argument("--sparse", type=bool_flag, default=False,
                            help="Perform sparse updates for the values")
        parser.add_argument("--k_dim", type=int, default=256,
                            help="Memory keys dimension")
        parser.add_argument("--heads", type=int, default=4,
                            help="Number of memory heads")
        parser.add_argument("--knn", type=int, default=32,
                            help="Number of memory slots to read / update - k-NN to the query")
        parser.add_argument("--n_keys", type=int, default=512,
                            help="Number of keys")
        parser.add_argument("--query_batchnorm", type=bool_flag, default=False,
                            help="Query MLP batch norm")

        # dropout
        parser.add_argument("--input_dropout", type=float, default=0,
                            help="Input dropout")
        parser.add_argument("--query_dropout", type=float, default=0,
                            help="Query dropout")
        parser.add_argument("--value_dropout", type=float, default=0,
                            help="Value dropout")


class Net(nn.Module):
    def __init__(self, pkm_on=True):
        super(Net, self).__init__()
        
        params = AttrDict({
            "sparse": False,
            "k_dim": 128,
            "heads": 4,
            "knn": 32,
            "n_keys": 512,  # the memory will have (n_keys ** 2) values
            "query_batchnorm": True,
            "input_dropout": 0,
            "query_dropout": 0,
            "value_dropout": 0,
        })

        self.memory_on = pkm_on
        
        
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)

        if self.memory_on:
            self.memory = HashingMemory(128, 128, params)
        else:
            self.memory = nn.Linear(128,128)

        self.fc2 = nn.Linear(128, 10)



    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = x + self.memory(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer1, optimizer2, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--pkm_off', default=False, action="store_true", help='turn off pkm memory')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net(not args.pkm_off).to(device)
    optimizer1 = optim.Adadelta([{"params": model.conv1.parameters()},
                                {"params": model.conv2.parameters()},
                                {"params": model.fc1.parameters()},
                                {"params": model.fc2.parameters()},], lr=args.lr)
    # facebook recommends a second optimizer for the memory layer
    optimizer2 = optim.Adam(model.memory.parameters(), lr=1e-3)

    scheduler = StepLR(optimizer1, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer1, optimizer2, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()