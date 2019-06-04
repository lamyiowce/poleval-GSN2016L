import argparse
import datetime
import torch
from torch import optim, nn
from torch.utils.data import random_split
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from models import WordEmbedder, LanguageModel, Classifier
from utils import MaskedSentencesDataset, ALPHABET


def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()
    total_loss = 0
    n = 0
    for batch_idx, (sentence, word, target) in enumerate(train_loader):
        # sentence, word, target = sentence.to(device), word.to(device), target.to(device)
        sentence, word, target = sentence, word, target
        optimizer.zero_grad()

        # loss = F.mse_loss(x, torch.zeros_like(x))
        output = model(sentence, word)
        loss = F.nll_loss(output, target.long())

        total_loss += loss
        n += 1
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 9 or batch_idx == len(train_loader.dataset) - 1:
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch, batch_idx * len(target), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    total_loss /= n
    # writer.add_scalar("train_loss", total_loss.item(), epoch)


def test(model, device, test_loader, writer, epoch):
    model.eval()
    test_loss = 0
    n = 0
    correct = 0
    print('[{}] Starting test evaluation.'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    with torch.no_grad():
        for batch_idx, (sentence, word, target) in enumerate(test_loader):
            sentence, word, target = sentence.to(device), word.to(device), target.to(device)

            output = model(sentence, word)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            n += 1

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).float().mean().item()

    test_loss /= n
    test_acc = 100. * correct / n

    # writer.add_scalar("test_loss", test_loss, epoch)
    # writer.add_scalar("test_acc", test_acc, epoch)
    print('[{}] Test set: Average loss: {:.4f}, Accuracy: ({:.2f}%)'.format(
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        test_loss, test_acc))

    return test_acc


def main():
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description='Language model training')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--test-batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train-data', default='./data/train.pkl', type=str)
    parser.add_argument('--test-data', default='./data/test.pkl', type=str)
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()
    use_cuda = args.cuda

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {}# {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    print("Using device: " + str(device))

    # Training settings.
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    train_data_path = args.train_data
    test_data_path = args.test_data
    use_cache = not args.no_cache

    print("Loading training dataset from " + train_data_path)
    train_dataset = MaskedSentencesDataset(train_data_path, device, cache=use_cache)

    print("Loading test dataset from " + test_data_path)
    test_dataset = MaskedSentencesDataset(test_data_path, device, cache=use_cache)

    print("Dataset loaded.")

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = torch.nn.DataParallel(model)
    #     batch_size = batch_size * torch.cuda.device_count()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)

    best_acc = -1

    model = LanguageModel(n_letters=len(ALPHABET) + 1, word_embed_dim=100, sentence_embed_dim=200,
                          classifier_layers=[128, 16]).to(device)
    print(model)

    writer = None  # SummaryWriter()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, writer)
        acc = test(model, device, test_loader, writer, epoch)
        if acc > best_acc:
            torch.save(model, "lang_model_{:.2f}.pt".format(acc))
            best_acc = acc

    print("\n*** BEST ACCURACY IS ", best_acc)


if __name__ == '__main__':
    main()
