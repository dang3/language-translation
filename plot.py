
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



labels = ['Default', 'Reverse off', '4 layers', '256 state size', '30k vocab', '20 epochs']

bleu_score_train_avg_short = [0.337693429, 0.337693429, 0.328676128, 0.336560669, 0.36819515, 0.410511144]
bleu_score_new_avg_short = [0.255864046, 0.255863969, 0.240023245, 0.314597871, 0.273991031, 0.280613459]

bleu_score_train_avg_medium = [0.105207364, 0.105207364, 0.163477231, 0.182310444, 0.183628119, 0.177556323]
bleu_score_new_avg_medium = [0.085402853, 0.085402853, 0.111527532, 0.159654115, 0.134335144, 0.114917442]

bleu_score_train_avg_long = [0.064351029, 0.064351029, 0.058159589, 0.144066862, 0.157092787, 0.086867458]
bleu_score_new_avg_long = [0.076232168, 0.076232168, 0.052071328, 0.145009693, 0.146557236, 0.05584472]


bleu_overallAverage = [0.154125148, 0.154125135, 0.158989176, 0.213699942, 0.210633244, 0.187718424]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, bleu_score_train_avg_medium, width, label='Train Data - Medium Sentences')
rects2 = ax.bar(x + width/2, bleu_score_new_avg_medium, width, label='New Data - Medium Sentences')




ax.set_ylabel('Average BLEU Scores')
ax.set_title('Average BLEU Scores for Medium Sentences')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()

plt.show()

