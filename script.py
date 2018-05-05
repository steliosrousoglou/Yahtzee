lines_seen = set() # holds lines already seen
outfile = open("training.data", "w")
for line in open("train.dat", "r"):
    if line not in lines_seen: # not a duplicate
        outfile.write(line)
        lines_seen.add(line)
outfile.close()
