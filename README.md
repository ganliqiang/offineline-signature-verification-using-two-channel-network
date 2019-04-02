# offineline-signature-verification-using-two-channel-network
the  implementation code that can reach more than  80% on GPDS960dataset

command:
$ python SigNet_v1.py --dataset <datase_name>

results:
1598/1600 [============================>.] - ETA: 0s
1599/1600 [============================>.] - ETA: 0s
1600/1600 [==============================] - 450s 281ms/step
* Accuracy on training set: 86.33%
* Accuracy on test set: 83.25%
i run 15epoch and use all GPDS960 datasets,all parameter are used deaulte in keras. you can continue finetune to get more accurate 
result. i am just for fun to accomplish this work.


NOte:
please feel free to let me know if you have any questions


reference:
1 https://github.com/sounakdey/SigNet
