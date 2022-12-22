import pickle

with open('loss_acc.pickle', 'rb') as f:
    loss_acc = pickle.load(f)

print(loss_acc)