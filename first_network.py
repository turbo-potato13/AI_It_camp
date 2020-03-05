import numpy as np

cake=0.0
rain=1.0
friend=0.0

def activation_function(x):
    if x>=0.5:
        return 1
    else:
        return 0

def predict(cake, rain, friend):
    inputs=np.array([cake, rain, friend])
    weights_hid_1=[0.25, 0.25, 0]
    weights_hid_2 = [0.5, -0.4, 0.9]
    weights_hid=np.array([weights_hid_1, weights_hid_2])

    weights_hid_out=np.array([-1,1])

    hid_in=np.dot(weights_hid, inputs)
    print('hid_in: ' +str(hid_in))

    hid_out = np.array([activation_function(x) for x in hid_in])
    print('hid_out: ' + str(hid_out))

    out=np.dot(weights_hid_out, hid_out)
    print("out: " +str(out))
    return activation_function(out)==1

print("result: " +str(predict(cake,rain,friend)))
