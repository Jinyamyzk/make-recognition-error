import pickle

with open('loss_acc.pickle', 'rb') as f:
    loss_acc = pickle.load(f)
loss_acc["train_acc"] = [acc.tolist()[0] for acc in loss_acc["train_acc"]]
loss_acc["valid_acc"] = [acc.tolist()[0] for acc in loss_acc["valid_acc"]]
print(loss_acc)