import numpy as np
import subprocess 
bidirection = 1
for activation in ["None", "nn.ReLU", "nn.Tanh"]:
    for sr in np.linspace(0.1, 2.0, 8):
        command = "python -u randsent.py --model esn --pooling max --pos_enc 0 --output_dim 2048 --zero 0 --spectral_radius {0} --leaky 0 --concat_inp 0 --stdv 0.3 --activation {2} --task_type probing --bidirectional {3} --sparsity 0.3 --gpu 0 --out-path result/SR_leakly0_bidi{3}_probing_act_{2}/result_sr_probing_{1:s}_act_{2}.json".format(sr, str(sr)[:5].replace(".","p"), activation, bidirection)
        subprocess.call(command, shell=True)
        print("end SR={}, activation={}".format(sr, activation) )