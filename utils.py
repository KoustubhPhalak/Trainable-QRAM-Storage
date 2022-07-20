import torch

def print_and_save(s, fname):
    print(s)
    with open(fname,"a") as f:
        f.write(s + "\n")

def hamming_distance(s1, s2, data_lines):
    s1 = s1.detach().numpy()
    s2 = s2.detach().numpy()
    hd = torch.zeros(s1.shape[0])
    for i in range(s1.shape[0]):
        n1 = int(s1[i].item())
        n2 = int(s2[i].item())
        xor_val = n1 ^ n2
        while xor_val > 0:
            hd[i] += xor_val & 1
            xor_val >>= 1
    return hd

def smallest_power_2(n):
    cnt = 0
    while n > int(2**cnt):
        cnt += 1
    return int(2**cnt)

def bin_to_decimal(data):
    data = torch.flip(data,[0])
    val = 0
    for i in range(data.shape[0]):
        val += data[i] * (2**i)
    return val
