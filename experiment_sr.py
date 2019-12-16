import numpy as np
import subprocess 
for sr in np.linspace(0.1, 2.0, 10):
    command = "python -u randsent.py --model esn --pooling mean --pos_enc 0 --output_dim 2048 --zero 1 --spectral_radius {} --leaky 0 --concat_inp 0 --stdv 0.1 --activation None --bidirectional 0 --sparsity 0.5 --gpu 0 --out-path result/SR_leakly0_o1di/result_sr_{:s}.json".format(sr, str(sr)[:5].replace(".","p"))
    subprocess.call(command, shell=True)
    print("end SR={}".format(sr) )