# Repo for this Application
GitHubRepo='https://github.com/jamdoran/scanned_film_alignment'


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


# iMessage Settings
iMessageAppleScript     = 'iMessage.scpt'                   # iMessage Applescript
appName                 = f'Scanned Film Alignment'         # Application Name shown in message
iMessageSubscriber      = 'jimmy@the-dorans.com'            # iMessage Subscriber to receive notification


# Jpeg Quality for adjusted files
jpeg_quality = 95


# Default Initial Folder
# Some examples below...  Just replace the active setting with one of these
# 
# default_folder = '/Volumes/'          #  Where all the external volumes are mounted (all volumes really)
# default_folder = '/Volumes/SSD/'      #  A specific Volume called SSD
# default_folder = '~'                  #  My Home Folder
# default_folder = '.'                  #  The same folder as the script
default_folder = '.'      