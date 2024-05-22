import matplotlib.pyplot as plt

f_dir = './logs/train_log.log'

f = open(f_dir,'r')
train_loss = []
valid_loss = []
for line in f:
    line=line.replace('\n','')
    lst = line.split(' ')
    train_loss.append(float(lst[0]))
    valid_loss.append(float(lst[1]))
f.close()

x = []
for i in range(len(train_loss)):
    x.append(i+1)

plt.plot(x,train_loss,label="train")
plt.plot(x,valid_loss,label="valid")
plt.legend()
plt.ylim(-0.2,2)
plt.show()