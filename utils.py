def print_and_save(s, fname):
    print(s)
    with open(fname,"a") as f:
        f.write(s + "\n")