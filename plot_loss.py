import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('losses_record.pkl', 'rb') as f:
    losses = pickle.load(f)


losses_sort = sorted(losses.items(),key=lambda x:x[0])
x = [item[0] for item in losses_sort]
y = [item[1] for item in losses_sort]
y = np.asarray(y)
rpn_loc_loss = y[:,0]
rpn_cls_loss = y[:,1]
roi_loc_loss = y[:,2]
roi_cls_loss = y[:,3]
total_loss = y[:,4]

window_smooth = 10

sx = []
srpn_loc_loss = []
srpn_cls_loss = []
sroi_loc_loss = []
sroi_cls_loss = []
stotal_loss = []
for i in range(len(x)//window_smooth):
    sx.append(x[i*window_smooth])
    srpn_loc_loss.append(rpn_loc_loss[i*window_smooth:(i+1)*window_smooth].mean())
    srpn_cls_loss.append(rpn_cls_loss[i * window_smooth:(i + 1) * window_smooth].mean())
    sroi_loc_loss.append(roi_loc_loss[i * window_smooth:(i + 1) * window_smooth].mean())
    sroi_cls_loss.append(roi_cls_loss[i * window_smooth:(i + 1) * window_smooth].mean())
    stotal_loss.append(total_loss[i * window_smooth:(i + 1) * window_smooth].mean())

plt.plot(sx,srpn_loc_loss,'r')
plt.plot(sx,srpn_cls_loss,'g')
plt.plot(sx,sroi_loc_loss,'b')
plt.plot(sx,sroi_cls_loss,'k')
plt.plot(sx,stotal_loss)

plt.show()
