import numpy as np
import subprocess 
bidirection = 1
for activation in ["nn.ReLU", "nn.Tanh", "None"]:
    for sr in np.linspace(0.1, 2.0, 8):
        acstr= activation.replace("nn.", "")
        srstr = str(sr)[:5].replace(".","p")
        command = f"python -u randsent.py --model esn --pooling max --pos_enc 0 --output_dim 2048 --zero 0 --spectral_radius {sr} --leaky 0 --concat_inp 0 --stdv 0.3 --activation {activation} --task_type probing --bidirectional {bidirection} --sparsity 0.3 --gpu 0 --out-path result/SR_leakly0_bidi{bidirection}_probing_act_{acstr}/result_sr_probing_{acstr}_{srstr}.json"
        subprocess.call(command, shell=True)
        print("end SR={}, activation={}".format(sr, activation) )