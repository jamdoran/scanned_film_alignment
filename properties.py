
# Settings and other stuff

# Where to store the adjusted files - this is a subfolder of the main folder
output_folder = 'Sprocket_Adjusted'

# Set to True if you want the program to align horizontally as well as vertically (vertically is default and cannot be turned off)
# Has no real performance impact
Horizontal = True

# Set to less than 1 for unrestricted
# Note that unrestricted means cpu_count -1 otherwise the machine may become unresponsive or crash since Seb won't upgrade his kit
# Setting it higher than the actual core count will result in it using cpu_count -1 anyways
# Setting it to between 1 and cpu_count -1 will result in that number of cores being used
# If the number of files is <50, it will just default to one as the convert only take a few seconds on one core
# if this value is not set, defaults to cpu_max -1
thread_max = -1



GitHubRepo='https://github.com/jamdoran/scanned_film_alignment'
