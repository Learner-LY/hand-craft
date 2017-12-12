import numpy as np
import h5py
from skimage import io
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim as optim

fil=h5py.File('/ly/shapemodel.h5','r')
fly=open('/ly/testface_landmarks.txt')
ftxt=fly.readlines()
for i in range(len(ftxt)):
    ftxt[i]=ftxt[i].strip('\n').split(' ')
for i in range(68):
    for j in range(2):
        ftxt[i][j]=float(ftxt[i][j])

U_label=Variable(torch.FloatTensor(np.array(ftxt).T))
d=fil['keypoints'][:][0]
sigma=torch.FloatTensor(fil['sigma'][:])#50*1;
sigma=(sigma-torch.mean(sigma))/torch.std(sigma)
sigma=Variable(sigma)
s_mean_raw=torch.FloatTensor(fil['mean_shape'][:])  #159645*1;
s_mean_raw=(s_mean_raw-torch.mean(s_mean_raw))/torch.std(s_mean_raw)
s_raw=torch.FloatTensor(fil['pca_basis'][:])   #159645*50;
for i in range(50):
    s_raw[:,i]=(s_raw[:,i]-torch.mean(s_raw[:,i]))/torch.std(s_raw[:,i])
s_mean=torch.rand(204,1)
s=torch.rand((204,50))
for i in range(204):
    if i%3==0:
        s_mean[i]=s_mean_raw[d[i/3]*3]
        s_mean[i+1]=s_mean_raw[d[i/3]*3+1]
        s_mean[i+2]=s_mean_raw[d[i/3]*3+2]
for i in range(50):
    for j in range(204):
        if j%3==0:
            s[j,i]=s_raw[d[j/3]*3,i]
            s[j+1,i]=s_raw[d[j/3]*3+1,i]
            s[j+2,i]=s_raw[d[j/3]*3+2,i]
s=Variable(s)
s_mean=Variable(s_mean)
P=Variable(torch.FloatTensor([[1,0,0],[0,1,0]]))
alpha=Variable(torch.rand((50,1)),requires_grad=True)
S=s_mean+s.mm(alpha)
S=torch.t(S.view(68,3))
R=Variable(torch.rand((3,3)),requires_grad=True)
t=Variable(torch.rand((2)),requires_grad=True)
U_proj=P.mm(R).mm(S)
for i in range(68):
    U_proj[:,i]=U_proj[:,i]+t

optimizer = optim.SGD([R,t,alpha], lr = 0.0005)
#optimizer1 = optim.Adam([R,t,alpha], lr = 0.000000001)

loss=((alpha*alpha)/(sigma*sigma)).mean()+((U_proj-U_label)*(U_proj-U_label)).mean()
loss.backward()
optimizer.step()

loss_print=[]
loss_1000=0

for i in range(50000):
    optimizer.zero_grad()
    alpha=alpha
    S=s_mean+s.mm(alpha)
    S=torch.t(S.view(68,3))
    R=R
    t=t
    U_proj=P.mm(R).mm(S)
    for ii in range(68):
        U_proj[:,ii]=U_proj[:,ii]+t
    loss=((alpha*alpha)/(sigma*sigma)).mean()+((U_proj-U_label)*(U_proj-U_label)).mean()
    loss_1000=loss_1000+loss.data[0]
    #print;
    loss.backward()
    optimizer.step()
    if i%1000==0:
        loss_print.append(loss_1000/1000.0)
        #print(t.data,R.data,alpha.data[0],loss_100/100.0)
	print(i,loss_1000/1000.0)
	loss_1000=0
    if i==10000:
	optimizer = optim.SGD([R,t,alpha], lr = 0.0002)
    if i==20000:
  	optimizer = optim.SGD([R,t,alpha], lr = 0.0001)
    #if i==30000:
	#optimizer = optim.SGD([R,t,alpha], lr = 0.00005)
    #if i==25000:
	#optimizer = optim.SGD([R,t,alpha], lr = 0.000005)
    #if i==30000:
	#optimizer = optim.SGD([R,t,alpha], lr = 0.000001)

np.save('/output/f.npy',t.data.numpy())
np.save('/output/R.npy',R.data.numpy())
np.save('/output/alpha.npy',alpha.data.numpy())
np.save('/output/loss.npy',np.array(loss_print))
np.save('/output/U.npy',U_proj.data.numpy())
