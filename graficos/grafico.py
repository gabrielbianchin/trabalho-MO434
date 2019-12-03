import matplotlib
import matplotlib.pyplot as plt

enet_cross_test = {'soil_acc': 0.996, 'soil_pre': 0.998, 'soil_rec': 0.998, 'soil_f1': 0.998, 'crop_acc': 0.998, 
                   'crop_pre': 0.917, 'crop_rec': 0.928, 'crop_f1': 0.921, 'weed_acc': 0.998, 'weed_pre': 0.724,
                   'weed_rec': 0.563, 'weed_f1': 0.620, 'fps': 47}
enet_focal_test = {'soil_acc': 0.996, 'soil_pre': 0.998, 'soil_rec': 0.998, 'soil_f1': 0.998, 'crop_acc': 0.998, 
                   'crop_pre': 0.911, 'crop_rec': 0.931, 'crop_f1': 0.920, 'weed_acc': 0.998, 'weed_pre': 0.720, 
                   'weed_rec': 0.575, 'weed_f1': 0.623, 'fps': 51}
enet_cross_test_3000 = {'soil_acc': 0.994, 'soil_pre': 0.995, 'soil_rec': 0.998, 'soil_f1': 0.997, 'crop_acc': 0.996, 
                   'crop_pre': 0.853, 'crop_rec': 0.876, 'crop_f1': 0.861, 'weed_acc': 0.996, 'weed_pre': 0,
                   'weed_rec': 0, 'weed_f1': 0, 'fps': 58}
enet_focal_test_3000 = {'soil_acc': 0.993, 'soil_pre': 0.996, 'soil_rec': 0.997, 'soil_f1': 0.997, 'crop_acc': 0.994, 
                   'crop_pre': 0.781, 'crop_rec': 0.869, 'crop_f1': 0.818, 'weed_acc': 0.996, 'weed_pre': 0.034,
                   'weed_rec': 0, 'weed_f1': 0, 'fps': 58}
psp_cross_test_resnet101 = {'soil_acc': 0.991, 'soil_pre': 0.996, 'soil_rec': 0.994, 'soil_f1': 0.995, 'crop_acc': 0.994, 
                  'crop_pre': 0.744, 'crop_rec': 0.561, 'crop_f1': 0.599, 'weed_acc': 0.991, 'weed_pre': 0.366, 
                  'weed_rec': 0.174, 'weed_f1': 0.205, 'fps': 11}
psp_cross_test = {'soil_acc': 0.993, 'soil_pre': 0.995, 'soil_rec': 0.998, 'soil_f1': 0.996, 'crop_acc': 0.994, 
                  'crop_pre': 0.795, 'crop_rec': 0.748, 'crop_f1': 0.739, 'weed_acc': 0.997, 'weed_pre': 0.459, 
                  'weed_rec': 0.265, 'weed_f1': 0.300, 'fps': 22}
psp_cross_test_pretrain = {'soil_acc': 0.996, 'soil_pre': 0.998, 'soil_rec': 0.997, 'soil_f1': 0.998, 'crop_acc': 0.997, 
                  'crop_pre': 0.844, 'crop_rec': 0.914, 'crop_f1': 0.875, 'weed_acc': 0.998, 'weed_pre': 0.622, 
                  'weed_rec': 0.401, 'weed_f1': 0.463, 'fps': 11}
psp_focal_test = {'soil_acc': 0.994, 'soil_pre': 0.995, 'soil_rec': 0.998, 'soil_f1': 0.997, 'crop_acc': 0.994,
                  'crop_pre': 0.831, 'crop_rec': 0.732, 'crop_f1': 0.749, 'weed_acc': 0.995, 'weed_pre': 0.473, 
                  'weed_rec': 0.256, 'weed_f1': 0.297, 'fps': 23}
psp_cross_test_3000 = {'soil_acc': 0.978, 'soil_pre': 0.995, 'soil_rec': 0.982, 'soil_f1': 0.987, 'crop_acc': 0.989, 
                  'crop_pre': 0.684, 'crop_rec': 0.543, 'crop_f1': 0.568, 'weed_acc': 0.987, 'weed_pre': 0.148, 
                  'weed_rec': 0.102, 'weed_f1': 0.099, 'fps': 22}
psp_focal_test_3000 = {'soil_acc': 0.977, 'soil_pre': 0.991, 'soil_rec': 0.985, 'soil_f1': 0.987, 'crop_acc': 0.988, 
                  'crop_pre': 0.542, 'crop_rec': 0.354, 'crop_f1': 0.392, 'weed_acc': 0.983, 'weed_pre': 0.018, 
                  'weed_rec': 0.004, 'weed_f1': 0.003, 'fps': 22}
unet_cross_test = {'soil_acc': 0.988, 'soil_pre': 0.999, 'soil_rec': 0.988, 'soil_f1': 0.993, 'crop_acc': 0.997, 
                  'crop_pre': 0.899, 'crop_rec': 0.918, 'crop_f1': 0.903, 'weed_acc': 0.989, 'weed_pre': 0.631, 
                   'weed_rec': 0.546, 'weed_f1': 0.545, 'fps': 24}
unet_focal_test = {'soil_acc': 0.987, 'soil_pre': 0.999, 'soil_rec': 0.987, 'soil_f1': 0.992, 'crop_acc': 0.997, 
                   'crop_pre': 0.918, 'crop_rec': 0.918, 'crop_f1': 0.913, 'weed_acc': 0.988, 'weed_pre': 0.644, 
                   'weed_rec': 0.633, 'weed_f1':0.599, 'fps': 24}
unet_cross_test_3000 = {'soil_acc': 0.988, 'soil_pre': 0.997, 'soil_rec': 0.990, 'soil_f1': 0.994, 'crop_acc': 0.988, 
                  'crop_pre': 0.726, 'crop_rec': 0.794, 'crop_f1': 0.706, 'weed_acc': 0.996, 'weed_pre': 0.001, 
                   'weed_rec': 0, 'weed_f1': 0, 'fps': 24}
unet_focal_test_3000 = {'soil_acc': 0.992, 'soil_pre': 0.996, 'soil_rec': 0.995, 'soil_f1': 0.996, 'crop_acc': 0.992, 
                   'crop_pre': 0.807, 'crop_rec': 0.745, 'crop_f1': 0.737, 'weed_acc': 0.996, 'weed_pre': 0.121, 
                   'weed_rec': 0.001, 'weed_f1':0.002, 'fps': 23}
bisenet_cross_test_resnet18 = {'soil_acc': 0.996, 'soil_pre': 0.997, 'soil_rec': 0.998, 'soil_f1': 0.998, 'crop_acc': 0.997, 
                   'crop_pre': 0.890, 'crop_rec': 0.923, 'crop_f1': 0.905, 'weed_acc': 0.998, 'weed_pre': 0.758, 
                   'weed_rec': 0.342, 'weed_f1':0.444, 'fps': 79}
bisenet_cross_test_resnet101 = {'soil_acc': 0.994, 'soil_pre': 0.996, 'soil_rec': 0.998, 'soil_f1': 0.997, 'crop_acc': 0.996, 
                   'crop_pre': 0.832, 'crop_rec': 0.908, 'crop_f1': 0.864, 'weed_acc': 0.996, 'weed_pre': 0.286, 
                   'weed_rec': 0.001, 'weed_f1':0.002, 'fps': 24}
uppernet_cross_test = {'soil_acc': 0.990, 'soil_pre': 0.997, 'soil_rec': 0.992, 'soil_f1': 0.995, 'crop_acc': 0.994, 
                   'crop_pre': 0.786, 'crop_rec': 0.801, 'crop_f1': 0.769, 'weed_acc': 0.995, 'weed_pre': 0.392, 
                   'weed_rec': 0.148, 'weed_f1': 0.186, 'fps': 25}
uppernet_focal_test = {'soil_acc': 0.991, 'soil_pre': 0.997, 'soil_rec': 0.994, 'soil_f1': 0.995, 'crop_acc': 0.994, 
                   'crop_pre': 0.779, 'crop_rec': 0.825, 'crop_f1': 0.787, 'weed_acc': 0.995, 'weed_pre': 0.483, 
                   'weed_rec': 0.207, 'weed_f1': 0.253, 'fps': 13}
uppernet_cross_test_3000 = {'soil_acc': 0.987, 'soil_pre': 0.994, 'soil_rec': 0.992, 'soil_f1': 0.993, 'crop_acc': 0.988, 
                   'crop_pre': 0.716, 'crop_rec': 0.556, 'crop_f1': 0.587, 'weed_acc': 0.996, 'weed_pre': 0.026, 
                   'weed_rec': 0, 'weed_f1':0, 'fps': 23}
uppernet_focal_test_3000 = {'soil_acc': 0.987, 'soil_pre': 0.994, 'soil_rec': 0.992, 'soil_f1': 0.993, 'crop_acc': 0.988, 
                   'crop_pre': 0.716, 'crop_rec': 0.556, 'crop_f1': 0.587, 'weed_acc': 0.996, 'weed_pre': 0.026, 
                   'weed_rec': 0, 'weed_f1':0, 'fps': 23}
upper_cross_test_pretrain = {'soil_acc': 0.996, 'soil_pre': 0.997, 'soil_rec': 0.999, 'soil_f1': 0.998, 'crop_acc': 0.997, 
                   'crop_pre': 0.917, 'crop_rec': 0.874, 'crop_f1': 0.890, 'weed_acc': 0.998, 'weed_pre': 0.687, 
                   'weed_rec': 0.423, 'weed_f1':0.492, 'fps': 25}

def dict_to_list(d):
    lista = []
    for key in d:
        lista.append(d[key])
    return lista

def media(d):
	acc = (d[0] + d[4] + d[8]) / 3.0
	pre = (d[1] + d[5] + d[9]) / 3.0
	rec = (d[2] + d[6] + d[10]) / 3.0
	f1 = (d[3] + d[7] + d[11]) / 3.0
	lista = []
	lista.append(acc)
	lista.append(pre)
	lista.append(rec)
	lista.append(f1)
	return lista

enet_cross = dict_to_list(enet_cross_test)
enet_cross_3000 = dict_to_list(enet_cross_test_3000)
enet_focal = dict_to_list(enet_focal_test)
enet_focal_3000 = dict_to_list(enet_focal_test_3000)
psp_cross = dict_to_list(psp_cross_test)
psp_cross_3000 = dict_to_list(psp_cross_test_3000)
psp_focal = dict_to_list(psp_focal_test)
psp_focal_3000 = dict_to_list(psp_focal_test_3000)
unet_cross = dict_to_list(unet_cross_test)
unet_cross_3000 = dict_to_list(unet_cross_test_3000)
unet_focal = dict_to_list(unet_focal_test)
unet_focal_3000 = dict_to_list(unet_focal_test_3000)
upper_cross = dict_to_list(uppernet_cross_test)
upper_cross_3000 = dict_to_list(uppernet_cross_test_3000)
upper_focal = dict_to_list(uppernet_focal_test)
upper_focal_3000 = dict_to_list(uppernet_focal_test_3000)
upper_cross_pre = dict_to_list(upper_cross_test_pretrain)
psp_cross_pretrain = dict_to_list(psp_cross_test_pretrain)

"""
enet_cross = media(enet_cross)
enet_cross_3000 = media(enet_cross_3000)
enet_focal = media(enet_focal)
enet_focal_3000 = media(enet_focal_3000)
psp_cross = media(psp_cross)
psp_cross_3000 = media(psp_cross_3000)
psp_focal = media(psp_focal)
psp_focal_3000 = media(psp_focal_3000)
unet_cross = media(unet_cross)
unet_cross_3000 = media(unet_cross_3000)
unet_focal = media(unet_focal)
unet_focal_3000 = media(unet_focal_3000)
upper_cross = media(upper_cross)
upper_cross_3000 = media(upper_cross_3000)
upper_focal = media(upper_focal)
upper_focal_3000 = media(upper_focal_3000)
"""


"""
enet_cross = media(enet_cross)
enet_focal = media(enet_focal)
psp_cross = media(psp_cross)
psp_focal = media(psp_focal)
unet_cross = media(unet_cross)
unet_focal = media(unet_focal)
bise_cross = media(bise_cross)
upper_cross = media(upper_cross)
upper_focal = media(upper_focal)

bise_renet18 = media(bise_renet18)
bise_resnet101 = media(bise_resnet101)
psp_resnet18 = media(psp_resnet18)
psp_resnet101 = media(psp_resnet101)
"""

psp_cross = media(psp_cross)
psp_cross_pretrain = media(psp_cross_pretrain)
upper_cross = media(upper_cross)
upper_cross_pre = media(upper_cross_pre)

width = 0.1

plt.figure(figsize = (10, 5))

r1 = [1,2,3,4]
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]
r4 = [x + width for x in r3]
r5 = [x + width for x in r4]
r6 = [x + width for x in r5]
r7 = [x + width for x in r6]
r8 = [x + width for x in r7]
r9 = [x + width for x in r8]
r10 = [x + width for x in r9]
r11 = [x + width for x in r10]
r12 = [x + width for x in r11]
r13 = [x + width for x in r12]
r14 = [x + width for x in r13]
r15 = [x + width for x in r14]
r16 = [x + width for x in r15]

"""
plt.bar(r1, enet_cross, width = width, label = 'E-net cross', color = '#010034')
plt.bar(r2, enet_cross_3000, width = width, label = 'E-net cross 3000', color = '#6dc5e5')
plt.bar(r3, enet_focal, width = width, label = 'E-net focal', color = '#0000fd')
plt.bar(r4, enet_focal_3000, width = width, label = 'E-net focal 3000', color = '#9f9cfd')
plt.bar(r5, unet_cross, width = width, label = 'U-net cross', color = '#064f04')
plt.bar(r6, unet_cross_3000, width = width, label = 'U-net cross 3000', color = '#6de26c')
plt.bar(r7, unet_focal, width = width, label = 'U-net focal', color = '#00ff00')
plt.bar(r8, unet_focal_3000, width = width, label = 'U-net focal 3000', color = '#99ff9b')
plt.bar(r9, psp_cross, width = width, label = 'PSPNet cross', color = '#6f0404')
plt.bar(r10, psp_cross_3000, width = width, label = 'PSPNet cross 3000', color = '#cf6e56')
plt.bar(r11, psp_focal, width = width, label = 'PSPNet focal', color = '#522622')
plt.bar(r12, psp_focal_3000, width = width, label = 'PSPNet focal 3000', color = '#ed2a27')
plt.bar(r13, upper_cross, width = width, label = 'Uppernet cross', color = '#FFA500')
plt.bar(r14, upper_cross_3000, width = width, label = 'Uppernet cross 3000', color = '#fffe27')
plt.bar(r15, upper_focal, width = width, label = 'Uppernet focal', color = '#ff8309')
plt.bar(r16, upper_focal_3000, width = width, label = 'Uppernet focal 3000', color = '#ffb659')

#plt.bar(r1, bise_renet18, width = width, label = 'Bisenet Resnet18', color = 'm')
#plt.bar(r2, bise_resnet101, width = width, label = 'Bisenet Resnet101', color = '#fe81ff')
#plt.bar(r3, psp_resnet18, width = width, label = 'PSPNet Resnet18', color = '#6f0404')
#plt.bar(r4, psp_resnet101, width = width, label = 'PSPNet Resnet101', color = '#cf6e56')

"""
plt.bar(r1, upper_cross, width = width, label = 'Uppernet', color = '#FFA500')
plt.bar(r2, upper_cross_pre, width = width, label = 'Uppernet Pretrain', color = '#fffe27')
plt.bar(r3, psp_cross, width = width, label = 'PSPNet', color = '#6f0404')
plt.bar(r4, psp_cross_pretrain, width = width, label = 'PSPNet Pretrain', color = '#cf6e56')

plt.xticks([r + 0.15 for r in range(1,5)], ['Acurácia', 'Precisão', 'Revocação', 'F1'])
plt.ylim(0, 1)
plt.title('Média das Classes')
plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.savefig('media_exp4.pdf', bbox_inches = 'tight')