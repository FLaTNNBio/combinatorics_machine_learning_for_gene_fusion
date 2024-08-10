import math

'''
dict_conflicts = {}
k = 3

input_fingerprint = 'training/fingerprint_CFL.txt'
input_fact = 'training/fact_fingerprint_CFL.txt'

fingerprint_file = open(input_fingerprint)
fingerprints = fingerprint_file.readlines()

fact_fingerprint_file = open(input_fact)
fact_fingerprints = fact_fingerprint_file.readlines()

count_k_fingers = 0
for i in range(0, len(fingerprints)):

    lengths = fingerprints[i]
    lengths_list = lengths.split()

    facts = fact_fingerprints[i]
    facts_list = facts.split()

    class_gene = lengths_list[0]
    class_gene = str(class_gene).replace('\n', '')

    lengths_list = lengths_list[1:]
    facts_list = facts_list[1:]

    for e in range(0, len(lengths_list[:-(k - 1)])):

        count_k_fingers += 1

        k_finger = lengths_list[e:e + k]
        k_finger_fact = facts_list[e:e + k]

        key = ''.join(i + ' '  for i in k_finger)
        key = key[:len(key)-1]

        key_string = ''.join(i for i in k_finger_fact)

        # Update dict_conflicts
        if key in dict_conflicts:
            key_dict = dict_conflicts[key]

            if key_string in key_dict:
                count_occurences = key_dict[key_string]
                count_occurences += 1
                key_dict[key_string] = count_occurences
            else:
                key_dict[key_string] = 1
        else:
            key_dict = {key_string:1}
            dict_conflicts[key] = key_dict

# Compute conflict
num_conflicts = 0
for key in dict_conflicts:
    key_dict = dict_conflicts[key]

    size_dict = len(key_dict)

    num_conflicts = num_conflicts + size_dict

num_keys = len(dict_conflicts)
print("# k_fingers: ", count_k_fingers)
print("# keys: ", num_keys)
print("# Conflicts: ", num_conflicts)
'''

file = open('longreads/experiment/30-september/sampled_read.fasta')
filtered_file = open('longreads/sampled_read_filtered.fasta', 'w')

lines = file.readlines()
filtered_lines = []

for i in range(len(lines)):

    if i == len(lines):
        break

    if i < len(lines)-1:
        if len(lines[i+1]) >= 5000:
            filtered_lines.append(lines[i])
            filtered_lines.append(lines[i+1])

    i = i+2

filtered_file.writelines(filtered_lines)
filtered_file.close()