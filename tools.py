import numpy as np
import matplotlib.pyplot as plt
from math import log

def plot_scores_from_logs(path: str):
    avg_scores, max_scores, stdev_scores = [], [], []
    with open(path) as f:
        for line in f:
            if ' - Iter ' in line:
                words = line.split()

                avg_scores.append(float(words[-6]))
                max_scores.append(int(words[-3]))
                stdev_scores.append(float(words[-1]))

    avg_scores = np.array(avg_scores)
    stdev_scores = np.array(stdev_scores)

    upper = np.add(avg_scores, stdev_scores)
    lower = np.add(avg_scores, -1*stdev_scores)

    plt.plot(np.arange(len(max_scores)), max_scores, label='Max Scores')
    plt.plot(np.arange(len(avg_scores)), avg_scores, label='Average Scores')
    plt.fill_between(np.arange(len(avg_scores)), avg_scores, upper, color='orange' , alpha = .5)
    plt.fill_between(np.arange(len(avg_scores)), avg_scores, lower, color='orange', alpha = .5)
    plt.xlabel('Iteration #')
    plt.ylabel('Score')
    plt.title('Score Over Time')
    plt.legend()
    plt.show()

def plot_scores_from_pretraining_logs(path: str):
    avg_scores, max_scores, stdev_scores = [], [], []
    with open(path) as f:
        for line in f:
            if 'Avg score:' in line:
                words = line.split()

                avg_scores.append(float(words[-6]))
                max_scores.append(int(words[-3]))
                stdev_scores.append(float(words[-1]))

    avg_scores = np.array(avg_scores)
    stdev_scores = np.array(stdev_scores)

    upper = np.add(avg_scores, stdev_scores)
    lower = np.add(avg_scores, -1*stdev_scores)

    plt.plot(np.arange(len(max_scores)), max_scores, label='Max Scores', color='blue')
    plt.plot(np.arange(len(avg_scores)), avg_scores, label='Average Scores', color='orange')
    plt.fill_between(np.arange(len(avg_scores)), avg_scores, upper, color='orange' , alpha = .5)
    plt.fill_between(np.arange(len(avg_scores)), avg_scores, lower, color='orange', alpha = .5)
    plt.xlabel('Iteration #')
    plt.ylabel('Score')
    plt.title('Score Over Time')
    plt.legend()
    plt.show()

def plot_loss_from_logs(path: str):
    train_loss, valid_loss = [], []
    with open(path) as f:
        for line in f:
            if 'epoch' in line and 'loss' in line and 'batch' in line:
                words = line.split()
                if 'train' in line:
                    train_loss.append(float(words[-1]))
                if 'valid' in line:
                    valid_loss.append(float(words[-1]))

    # fig, ax = plt.subplots(1, 2)

    plt.plot(np.arange(len(train_loss)), train_loss, label='Train Loss')
    plt.plot(np.linspace(start=0, stop=len(train_loss), num=len(valid_loss)), valid_loss, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (CSE)')
    plt.title('Loss Curve')
    plt.legend()

    # ax[1].plot(np.arange(len(train_loss)), [log(ell) for ell in train_loss], label='Train Loss')
    # ax[1].plot(np.linspace(start=0, stop=len(train_loss), num=len(valid_loss)), [log(ell) for ell in valid_loss], label='Validation Loss')
    # ax[1].xlabel('Iteration')
    # ax[1].ylabel('log(Loss (CSE))')
    # ax[1].title('Loss Curve')
    # ax[1].legend()

    plt.show()
