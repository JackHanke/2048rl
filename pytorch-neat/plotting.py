import matplotlib.pyplot as plt
from math import exp

def plot_training_from_logs(log_file: str):
    average_fitnesses, best_fitnesses, genome_lengths = [], [], []

    with open(log_file, 'r') as f:
        for line in f:
            if 'Best Genome Fitness' in line:
                best_fitnesses.append(exp(float(line.split()[-1])))
            elif 'Average Fitness' in line:
                average_fitnesses.append(exp(float(line.split()[-1])))
            elif 'Best Genome Length' in line:
                genome_lengths.append(float(line.split()[-1]))

    fig = plt.figure()
    ax = fig.add_subplot()


    
    # plt.plot([i+1 for i in range(len(average_fitnesses))], average_fitnesses, label='Average Fitness', color='#EDE0C8')
    # plt.plot([i+1 for i in range(len(best_fitnesses))], best_fitnesses, label='Best Fitness', color='#F59563')
    plt.plot([i+1 for i in range(len(genome_lengths))], genome_lengths, label='Genome Length', color='#F59563')

    axis_color = '#776E65'
    ax.xaxis.label.set_color(axis_color)
    ax.yaxis.label.set_color(axis_color)
    ax.tick_params(axis='x', colors=axis_color)
    ax.tick_params(axis='y', colors=axis_color)
    ax.spines['left'].set_color(axis_color)
    ax.spines['top'].set_color(axis_color)
    # ax.set_title(f'NEAT Score Curves', color=axis_color)
    ax.set_title(f'Genome Lengths Over Time', color=axis_color)

    plt.xlabel(f'Generations')
    plt.ylabel(f'Score')
    plt.ylabel(f'Genome Length')
    # plt.legend()
    # plt.savefig('../assets/fitness_curve.png', transparent=True)
    plt.savefig('../assets/genome_length_over_time.png', transparent=True)


if __name__ == '__main__':
    log_file = 'neat/experiments/gameof2048/2048.log'
    plot_training_from_logs(log_file=log_file)

