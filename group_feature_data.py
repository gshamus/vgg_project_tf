import numpy as np
import os
import sys

# Consants	
FEATURE_DIM = 4096

def group_npys(write_dir, read_dir, gender_label):
	file_paths = [os.path.join(read_dir, x) for x in os.listdir(read_dir)]

	result_feats = np.zeros((len(file_paths), FEATURE_DIM))
	result_labels = np.zeros((len(file_paths), 3))

	for i, path in enumerate(file_paths):
		age = int(os.path.basename(path).split('_')[0])
		result_feats[i, :] = np.load(path)
		if gender_label: # female
			result_labels[i, : ] = [0, 1, age]
		else:	   # male
			result_labels[i, : ] = [1, 0, age]

		if i % 500 == 0: print(i, gender_label)

	if gender_label:
		write_path = write_dir + 'all_female_'
	else:
		write_path = write_dir + 'all_male_'

	np.save(write_path + 'feats' +  '.npy', result_feats)
	np.save(write_path + 'labels' +  '.npy', result_labels)



def main():
	data_dir = './feature_data/'
	male_data_dir = './feature_data/male/' # [1, 0] label for males
	female_data_dir = './feature_data/female/' # [0, 1] label for females
	group_npys(data_dir, male_data_dir, 0)
	group_npys(data_dir, female_data_dir, 1)


if __name__ == '__main__':
	main()