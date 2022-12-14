import matplotlib.pyplot as plt
import pickle

with open('./images/pole-balancing-fitness.pkl', 'rb')as f:
    fitness = pickle.load(f)

plt.plot(fitness)
plt.xticks(ticks=range(0, len(fitness)+1, 5), labels=range(0, len(fitness)+1, 5))
plt.xlabel('Generations')
plt.xlim([0, 50])
plt.ylabel('Fitness')
plt.title('NEAT V2')
plt.grid()
plt.savefig('./images/plot')