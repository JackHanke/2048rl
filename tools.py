import numpy as np
import matplotlib.pyplot as plt

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

    plt.plot(np.arange(len(train_loss)), train_loss, label='Train Loss')
    plt.plot(np.linspace(start=0, stop=len(train_loss), num=len(valid_loss)), valid_loss, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (CSE)')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
