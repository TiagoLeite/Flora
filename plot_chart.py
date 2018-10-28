import seaborn as sea
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

data_100 = pd.read_csv('Results/1.0/5000_drop/run_validation-tag-accuracy_1.csv')
data_75 = pd.read_csv('Results/0.75/5000_drop/run_validation-tag-accuracy_1.csv')
data_50 = pd.read_csv('Results/0.50/5000_drop/run_validation-tag-accuracy_1.csv')
data_25 = pd.read_csv('Results/0.25/5000_drop/run_validation-tag-accuracy_1.csv')

data_100 = data_100.drop(columns=['Wall time'])
data_75 = data_75.drop(columns=['Wall time'])
data_50 = data_50.drop(columns=['Wall time'])
data_25 = data_25.drop(columns=['Wall time'])

# lines = data.plot.line(x='Step', y='Loss')
# lines = data2.plot.line(x='Step', y='Loss')
# plt.plot('Step', 'Loss', data=data, markerfacecolor='orange', color='orange', linewidth=2)
# plt.plot('Step', 'Loss', data=data2, markerfacecolor='blue', color='blue', linewidth=2)
# plt.show()

# log_var = tf.placeholder(shape=None, dtype=tf.float32)

log_var = tf.Variable(0.0)

init = tf.global_variables_initializer()
k = 0

losses_100 = data_100['Value']
steps_100 = data_100['Step']

losses_75 = data_75['Value']
steps_75 = data_75['Step']

losses_50 = data_50['Value']
steps_50 = data_50['Step']

losses_25 = data_25['Value']
steps_25 = data_25['Step']


print(len(losses_100))

tf.summary.scalar("loss", log_var)
merged = tf.summary.merge_all()

writer_100 = tf.summary.FileWriter('logs/100')
writer_75 = tf.summary.FileWriter('logs/75')
writer_50 = tf.summary.FileWriter('logs/50')
writer_25 = tf.summary.FileWriter('logs/25')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for step in range(len(losses_100)):
    # loop over several initializations of the variable
    summary = sess.run(merged, feed_dict={log_var: losses_100[step]})
    writer_100.add_summary(summary, global_step=steps_100[step])
    tf.summary.scalar('step', step)
    writer_100.flush()

for step in range(len(losses_75)):
    # loop over several initializations of the variable
    summary = sess.run(merged, feed_dict={log_var: losses_75[step]})
    writer_75.add_summary(summary, global_step=steps_75[step])
    tf.summary.scalar('step', step)
    writer_75.flush()

for step in range(len(losses_50)):
    # loop over several initializations of the variable
    summary = sess.run(merged, feed_dict={log_var: losses_50[step]})
    writer_50.add_summary(summary, global_step=steps_50[step])
    tf.summary.scalar('step', step)
    writer_50.flush()

for step in range(len(losses_25)):
    # loop over several initializations of the variable
    summary = sess.run(merged, feed_dict={log_var: losses_25[step]})
    writer_25.add_summary(summary, global_step=steps_25[step])
    tf.summary.scalar('step', step)
    writer_25.flush()



print('Done with writing the scalar summary')
