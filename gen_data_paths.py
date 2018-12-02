import os
import sys


def write_list_to_file(path_list, filename):
	with open(filename, 'w') as f:
		for path in path_list:
			f.write("%s\n" % path)

def main():
	data_dir = '../agegender_cleaned/combined/aligned/'

	all_sub_dirs = os.listdir(data_dir)
	male_paths = []
	female_paths = []
	for sub_dir in all_sub_dirs:
		data_sub_dir = os.path.join(data_dir, sub_dir)
		
		age = int(sub_dir.split('_')[0])
		gender = data_sub_dir[-1]

		data_file_names = os.listdir(data_sub_dir)
		data_paths = [os.path.join(data_sub_dir, x) + ' ' + str(age) for x in data_file_names if ' ' not in x]

		if gender == 'F':
			female_paths += data_paths
		elif gender == 'M':
			male_paths += data_paths


	write_list_to_file(female_paths, 'all_female_paths.txt')
	write_list_to_file(male_paths, 'all_male_paths.txt')



if __name__ == '__main__':
	main()