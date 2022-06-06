# Standard Library stuff
import os
import platform
from multiprocessing import Process, cpu_count
import subprocess
import shlex
import sys
import time
from tkinter import Tk
from tkinter import filedialog

# # pip installed modules
import numpy as np
from PIL import Image
import cv2
from numpy import array_split

# Local Imports
import screen
import properties as p

# List of folders where we do not want to be doing anything at all
# This is far from a comprehensive list
crazy_folders = ['/', '.Trash', '/tmp', '/Applications', '/Library', '/Volumes', '/System', '/dev', '/opt', '/private',
                 '/sbin', '/usr', '/Users']
text_line = ('_' * 100)


# Disk and file structure as follows:
#
#
# /root/                             (top level folder holding customer folders)
# /root/customer                     (customer folder below root)
# /root/customer/file.mov            (raw film scans in the customer folder - could be more than one)
# /root/customer/file_raw            (folder to hold jpegs from raw movie scans - folder is same name as movie with no suffix and _raw)
# /root/customer/file_adjusted       (folder to hold adjusted jpegs created by ffmpeg - folder is same name as raw movie with no suffix and _adjusted)
# /root/customer/file_adjusted.mov   (adjusted movie files - movie is stored in customer folder with same name as original with a _adjusted inseryed before suffix
#
# No other files are created
#


# All data and work is held and done here
# Driven from main() to make it easy to follow
class ImageProcessing:

    def __init__(self) -> None:

        # File and folder locations
        self.root_folder = None  # Root folder - contains customer folders with movies
        self.all_customer_folders = list()  # A list of all customer folders in the root folder
        self.all_raw_movies = list()  # A list of the full path to all raw movies in all customer folders

        # Working Images and Folders
        self.adjustment_list_SrcDest = list()  # A list of all extracted raw jpegs and the associated adjusted jpeg.  Chunked up and passed to adjustment worker process
        self.all_raw_jpeg_folders = list()  # A list of all folders containg raw jpegs extracted from the raw .mov files
        self.all_adjusted_jpeg_folders = list()  # A list of all folders containg adjusted jpegs

        # Chunked List of files to be processed (list of lists where len(list) is number of CPUs we can use)
        self.perProcessFileChunks = list()  # A list containing multiple lists of full-path src/dest [raw jpeg, adjusted jpeg] - passed to worker processes

        # Max number of CPU Cores we can use
        self.max_cpu_count = max_allowed_core_count()  # CPU cores we are allowed to use - see properties.py for more information

        # Store counters and timers
        self.raw_movie_count = 0  # A count of all the raw movies found in all customer folders
        self.all_raw_jpegs_count = 0  # A count of all raw jpegs extracted from all raw movies
        self.start_time = None  # Start time in time format
        self.start_time_text = None  # Start time in human format
        self.customer_folder_count = 0  # Count of customer folders inside the root folder

    # Get the root folder using TK.filedialog.askdirectory() and hide main window
    # Replaces wxPython due to its lang compile time on ARM64 architectures
    def getRootFolder(self) -> None:

        window = Tk()
        window.withdraw()  # Hide the tk window
        rootFolder = filedialog.askdirectory(initialdir=p.default_folder)
        window.destroy()

        # Check we got something
        if not rootFolder:
            print('User Cancelled !')
            exit()

        # Ensure we haven't selected any of the crazy folders
        if rootFolder in crazy_folders:
            print(f'Are you mad ? - you cant put files in {rootFolder} - pick a different folder\n')
            exit()

        # Make folder a full path
        rootFolder = os.path.join(rootFolder, '')
        self.root_folder = rootFolder

        # Start the stopwatch
        self.start_time = time.time()
        self.start_time_text = time.strftime("%I:%M:%S")

        return None

    # Get a list of customer folders in the root folder
    def get_customer_folders(self) -> None:
        subFolders = next(os.walk(self.root_folder))[1]
        subFolders = [os.path.join(self.root_folder, f, '') for f in subFolders]

        # Set counters
        self.all_customer_folders = sorted(subFolders)
        self.customer_folder_count = len(self.all_customer_folders)

        # if no customer folders then just exit
        if self.customer_folder_count == 0:
            print(f'\nFound no Customer Folders to process...\n\n')
            exit(1)
        else:
            print(f'Found {self.customer_folder_count} Customer Folders to process')
            for f in self.all_customer_folders: print(f' {f}')

    # Get a list() of every Mov or AVI file in any folder under the root folder (include the root folder)
    def get_all_movie_files(self) -> None:

        # Scan all customer folders and find all mov and avi files - if the working folders exist, then skip the folder
        for customerFolder in self.all_customer_folders:
            print(f'\n{screen.colour.BOLD}nProcessing movie files in {customerFolder} {screen.colour.END}')

            # Get all the files in the folder (no super easy way to do a complex filter here so do it step-by-step
            all_mov_files = next(os.walk(customerFolder))[2]
            # Extract just the .mov and .avi files
            all_mov_files = [f for f in all_mov_files if f.lower().endswith('.mov') or f.lower().endswith('.avi')]
            # Exclude all files that have already been processed
            all_mov_files = [f for f in all_mov_files if
                             not f.lower().endswith('_adjusted.mov') and not f.lower().endswith('_adjusted.avi')]
            # Append the full path to the remaining movie files
            all_mov_files = [os.path.join(customerFolder, f) for f in all_mov_files]
            # Count remaining files
            file_count = len(all_mov_files)

            print(f' Found {file_count} Raw Movies')

            # No need to do this stuff if there are no movies found
            if file_count > 0:
                print(f' Full list of Raw Movies :- ')
                for f in all_mov_files: print(f'   {f}')
                print()

                # Check if any of the raw files have already been processed
                for mov in all_mov_files:
                    jpeg_folder_raw = os.path.join(str(mov.split('.')[0]) + '_raw', '')
                    jpeg_folder_adjusted = os.path.join(str(mov.split('.')[0]) + '_adjusted', '')
                    adjusted_movie = str(mov.split('.')[0]) + '_adjusted.' + str(mov.split('.')[1])

                    print(f'Checking if {mov} has already been processed....')
                    print(f' Looking for {jpeg_folder_raw}')
                    print(f' Looking for {jpeg_folder_adjusted}')
                    print(f' Looking for {adjusted_movie}')

                    # Do not add the folder if the adjusted movie, or the _raw or _adjusted folders already exist
                    # Do not add the folder if the adjusted movie, or the _raw or _adjusted folders already exist
                    if os.path.isdir(jpeg_folder_raw) or os.path.isdir(jpeg_folder_adjusted) or os.path.exists(
                            adjusted_movie):
                        print(f' >> Evidence that Raw Movie {mov} has already been processed - skipping....')
                    else:
                        print(f' >> Adding {mov} to process')
                        self.all_raw_movies.append(mov)
                        self.raw_movie_count += 1

        # if no raw movie files then just exit
        if self.raw_movie_count == 0:
            print(f'{screen.colour.RED}Found no raw movie files to process...{screen.colour.END}\n\n')
            exit(1)

        # Sort the list of raw movies
        self.all_raw_movies.sort()

        # Show the final list of movies to be processed
        print(f'\n\n{text_line}')
        print(f'Total -- {self.raw_movie_count} raw movies to process')
        for f in self.all_raw_movies: print(f' > {f}')
        print(f' {text_line}')

    # Create a set of working folders to hold the raw and adjusted Jpeg files
    def create_working_folders(self) -> None:

        for mov in self.all_raw_movies:
            jpeg_folder_raw = os.path.join(mov.split('.')[0] + '_raw', '')
            jpeg_folder_adjusted = os.path.join(mov.split('.')[0] + '_adjusted', '')

            createFolder(jpeg_folder_raw)
            createFolder(jpeg_folder_adjusted)

            # Update the list of working folders
            self.all_raw_jpeg_folders.append(jpeg_folder_raw)
            self.all_adjusted_jpeg_folders.append(jpeg_folder_adjusted)

        self.all_raw_jpeg_folders.sort()
        self.all_adjusted_jpeg_folders.sort()

    # ffmpeg the movie into the _raw folder
    def extract_jpegs(self) -> None:

        # Process each movie with ffmpeg
        for mov in self.all_raw_movies:
            # Work out the path to the _rqw folder from the filename
            rawFolder = os.path.join(str(mov.split('.')[0]) + '_raw', '')
            cli = ffmpeg_split_command(mov, rawFolder)
            print(f'\n\n{screen.colour.BOLD}Processing Movie    : {mov}{screen.colour.END}')
            print(f'{screen.colour.BOLD} >> ffmpeg CLI      : {cli}{screen.colour.END}\n')
            run_ffmpeg(cli)

    # Get all the Jpeg Images from each Raw Folder into a list for adjustment
    # Double check for .jpg files in case any O/S junk has found its way in
    # Create a list containing [src_file, dst_file] for easy adjustment
    def gather_all_jpegs(self) -> None:
        for rawFolder, adjustFolder in zip(self.all_raw_jpeg_folders, self.all_adjusted_jpeg_folders):
            all_files = next(os.walk(rawFolder))[2]
            srcDestFiles = [[os.path.join(rawFolder, f), os.path.join(adjustFolder, f)] for f in all_files if
                            f.lower().endswith('.jpg')]
            self.adjustment_list_SrcDest.extend(srcDestFiles)

        # Sort files to keep some kind of order when processing
        self.adjustment_list_SrcDest.sort()

        self.all_raw_jpegs_count = len(self.adjustment_list_SrcDest)
        # TODO:  Just for debugging
        # for f in self.adjustment_list_SrcDest:
        #     print(f)
        # print (f'\n\n\n\n')
        # TODO:  Just for debugging

    # # Split the full list of files to be processed into chunks for hand off to worker processes
    # # Use numpy.array_split() - overkill, I know...
    def chunk_file_list(self) -> None:

        # If the number of raw jpeg files is less than 50, just use one process to do the work
        fileCount = len(self.adjustment_list_SrcDest)
        if fileCount < 50:
            self.perProcessFileChunks.append(self.adjustment_list_SrcDest)
        else:
            self.perProcessFileChunks = array_split(self.adjustment_list_SrcDest, self.max_cpu_count)

    # Create a bunch of sub-processes to adjust the raw jpeg files
    # Each process is passed a chunk of files to deal with
    def adjust_jpeg_files(self) -> None:

        # Create a list of sub processes
        subProcesses = list()

        for chunk in self.perProcessFileChunks:
            subProcesses.append(Process(target=adjustFile, args=(chunk,)))

        for proc in subProcesses:
            proc.start()

        for proc in subProcesses:
            proc.join()

        print(f'\nAll files adjusted!')

    # Use ffmpeg to create a new movie from the adjusted Jpeg Images
    def create_new_movie(self) -> None:
        print()
        print('All files adjusted!')
        print()
        print(f'Starting to combine {self.all_raw_jpegs_count} files into {self.raw_movie_count} adjusted movies')

        # Process each adjusted folder and create a new movie
        # Run a subprocess for each (one at a time)
        for mov in self.all_raw_movies:
            adjusted_movie = str(mov.split('.')[0]) + '_adjusted.' + str(mov.split('.')[1])
            adjusted_folder = os.path.join(str(mov.split('.')[0]) + '_adjusted', '')

            cli = ffmpeg_join_command(adjusted_folder, adjusted_movie)

            print(f'\n\n{screen.colour.BOLD}Creating new Movie    : {mov}{screen.colour.END}')
            print(f'{screen.colour.BOLD} >> ffmpeg CLI            : {cli}{screen.colour.END}\n')
            run_ffmpeg(cli)

    # Show a little summary at the end
    def summary(self) -> None:

        # Create a summary so we can print and send via iMessage
        l1 = f'Folder               : {self.root_folder}\n'
        l2 = f'Total Reels          : {self.raw_movie_count:,d}\n'
        l3 = f'Total Adjusted Files : {self.all_raw_jpegs_count:,d}\n'
        l4 = f'Start time           : {self.start_time_text}\n'
        l5 = f'End time             : {time.strftime("%I:%M:%S")}\n'
        l6 = f'Time to convert      : {time.time() - self.start_time:.2f} seconds\n'

        print()
        print(f'{screen.colour.YELLOW}- Summary -------------{screen.colour.END}\n')
        print(f'{l1}{l2}{l3}{l4}{l5}{l6}')
        print()

        # Try to send an iMessage, but just pass if it fails...
        try:
            message = f'{l1}{l2}{l3}{l4}{l5}{l6}'
            self.send_iMessage(p.iMessageSubscriber, message)
        except:
            pass


    # Try to send an imessage using AppleScript to drive iMessage
    # Note: First time you use this, the script will ask for permission for oascript to use iMessage
    def send_iMessage(self, iMessageSubscriber, message) -> None:
        # Create timestamps and message body
        timeStamp = time.strftime("%d/%m/%Y %H:%M:%S")
        messageBody = f'{p.appName} \n{timeStamp} \n\n{message}'

        # Send the message via iMessage
        iMessage_command = f'osascript {p.iMessageAppleScript} {iMessageSubscriber} "{messageBody}"'
        exitCode = os.system(iMessage_command)

        if exitCode != 0:
            print('iMessage Notification Failed!')

        print('iMessage Notification Sent OK')


def main():
    # Instantiate ImageProcessing object to store and process Images
    imageData = ImageProcessing()

    # Do some host checks
    hostInfo()

    # Get the root folder to process
    imageData.getRootFolder()

    # Build a list of all cusomter folders
    imageData.get_customer_folders()

    # Get all the raw movie files in all the customer folders
    imageData.get_all_movie_files()

    print(f'\nReady to Go.....\n')

    # Create working folders for the mov files
    imageData.create_working_folders()

    # ffmpeg all the raw movies into the _raw folder
    imageData.extract_jpegs()

    # Gather all the Jpeg files that were extracted
    imageData.gather_all_jpegs()

    # Chunk the data for passing to multiple processes
    imageData.chunk_file_list()

    # Adjust the jpeg files
    imageData.adjust_jpeg_files()

    # Create the new movie from the adjusted jpeg images @ 25 frames per second
    imageData.create_new_movie()

    # Show a summary and send an iMessage notificaiton if possible
    imageData.summary()


# Print host setting and allowed/configured CPU Cores that can be used
def hostInfo():
    print()
    print(screen.logo)
    print()

    if os.uname()[0] != 'Darwin':
        print(f'Only Supported on macOS - this is {os.uname()[0]}')
        exit(1)
    else:
        print(f'This is macOS version {platform.mac_ver()[0]}')

    print(f'{cpu_count()} CPU Cores detected')
    print(f'We can use maximum of {max_allowed_core_count()} Cores')
    print()
    check_ffmpeg_installed()
    print()


# Check ffmpeg is installed
def check_ffmpeg_installed():
    # Make sure ffmpeg is installed
    stream = os.popen('which ffmpeg')
    ffmpeg_location = stream.read().strip()

    if not ffmpeg_location:
        print(f'{screen.colour.RED}ffmpeg is NOT installed or is not on the $PATH {screen.colour.END}')
        exit(1)

    print(f'ffmpeg is installed ({ffmpeg_location})')


# -y            : overwrite without asking
# -i            : Source File
# -c            : codec
#  copy         : Stream copy is a mode selected by supplying the copy parameter to the -codec option.
#                 It makes ffmpeg omit the decoding and encoding step for the specified stream, so it does only demuxing and muxing.
#                 It is useful for changing the container format or modifying container-level metadata.
# -framerate    : Number of jpeg files to use per-second of movie
# frame_%d.jpg  : used for both export and import - export/import files with name Frame_nnn.jpg where nnn is a number sequence starting from 1.  No zero padding is used.

# Return the command line to run ffmpeg to split a movie into jpeg files
def ffmpeg_split_command(srcFile, dstFolder): return f'ffmpeg -y -i "{srcFile}" -c:v copy "{dstFolder}frame_%d.jpg"'


# Return the command line to run ffmpeg to join jpeg files into a mov file
def ffmpeg_join_command(adjusted_folder,
                        adjusted_movie): return f'ffmpeg -y -framerate 25 -i "{adjusted_folder}frame_%d.jpg" -codec copy "{adjusted_movie}"'


# Run the ffmpeg session in a subprocess and monitor stdOut
# Runs subprocess with Popen/poll to show live stdOut
def run_ffmpeg(cli, shellType=False, stdoutType=subprocess.PIPE) -> None:
    process = None

    try:
        process = subprocess.Popen(
            shlex.split(cli), shell=shellType, stdout=stdoutType)
    except:
        print(f'Error {sys.exc_info()[1]} while running {cli} ')
        exit()

    while True:
        output = process.stdout.readline()

        if process.poll() is not None:
            break
        if output:
            print(output.strip().decode())

        # process.poll()


# Create one folder and check it was created properly
def createFolder(folder):
    try:

        # Check the folder does not already exist
        if os.path.isdir(folder):
            print(f'Skipped Folder       : skipping {screen.colour.RED}{folder}{screen.colour.END} - it already exists')
            return False

        os.mkdir(folder)
        if not os.path.isdir(folder):
            print(f'Failed to create {folder} - the folder was not created, but no error was presented')
            print(f'This should not happen!')
            print(f'Stopping !')
            print()
            exit()

        print(f'Created Folder       : {folder}')

    except FileExistsError:
        print(f'Folder Create       : {folder} already exists')

    except OSError:
        print(f'{screen.colour.RED}Unable to create folders{screen.colour.END}')
        print(f'{screen.colour.RED}Probably no available space left on this volume{screen.colour.END}')
        print(f'{screen.colour.RED}Stopping - something is broken{screen.colour.END}')
        print()
        exit()


# Define the maximum allowed cores that this script can use
# Read entry from properties.py
def max_allowed_core_count():
    try:
        if p.thread_max < 1:  # Set to less than 0 means unrestricted so use cpu_max -1
            return (cpu_count() - 1)
        elif p.thread_max > cpu_count():  # Properties file is higher than number of cores in machine, so use cpu_max -1
            return (cpu_count() - 1)
        else:
            return (p.thread_max)  # Configured value is OK
    except:
        return (cpu_count() - 1)  # Likely not set in properties files, so set to cpu_max -1


## simple sprocket detection algorithm
## returns the detected sprocket position
## relative to vertical scan center and left horizontal edge
def detectSprocketPos(img, roi=[0.01, 0.09, 0.2, 0.8],  # region-of-interest - set as small as possible
                      thresholds=[0.5, 0.2],  # edge thresholds; first one higher, second one lower
                      filterSize=25,  # smoothing kernel - leave it untouched
                      minSize=0.1,  # min. relative sprocket size to be expected - used for checks
                      horizontal=p.Horizontal):  # if you want a simple horizontal alignment as well

    ## inital preparations

    # get the size of the image
    dy, dx, dz = img.shape

    # convert roi coords to real image coords
    x0 = int(roi[0] * dx)
    x1 = int(roi[1] * dx)

    y0 = int(roi[2] * dy)
    y1 = int(roi[3] * dy)

    # cutting out the strip to work with
    # we're using the full scan heigth to simplify computations
    sprocketStrip = img[:, x0:x1, :]

    # now calculating the vertical sobel edges
    sprocketEdges = np.absolute(cv2.Sobel(sprocketStrip, cv2.CV_64F, 0, 1, ksize=3))

    # by averaging horizontally, only promient horizontal edges
    # show up in the histogram
    histogram = np.mean(sprocketEdges, axis=(1, 2))

    # smoothing the histogram to make signal more stable.
    # sigma==0 -> it is autocalculated
    smoothedHisto = cv2.GaussianBlur(histogram, (1, filterSize), 0)

    # now analyzing the smoothed histogram
    # everything is relative to the detected maximum of the histogram
    # we only work in the region where the sprocket is expected
    maxPeakValue = smoothedHisto[y0:y1].max()

    # the outer threshold is used to search for high peaks from the outside
    # it should be as high as possible in order to suppress that the algorithm
    # locks onto bright imprints of the film stock
    outerThreshold = thresholds[0] * maxPeakValue

    # the inner threshold is used to really search for the boundaries of
    # the sprocket. Implicitly it is assumed here that the area within
    # the sprocket is very evenly lit. If a lot of dust is present in
    # the material, this threshold should be raised higher
    innerThreshold = thresholds[1] * maxPeakValue

    # searching for the sprocket from the outside, first from below
    # we start not right at the border of the histogram in order to
    # avoid locking at bad cuts which look like tiny sprockets
    # to the algorithm
    outerLow = y0
    for y in range(y0, y1):
        if smoothedHisto[y] > outerThreshold:
            outerLow = y
            break

    # now searching from above
    outerHigh = y1
    for y in range(y1, outerLow, -1):
        if smoothedHisto[y] > outerThreshold:
            outerHigh = y
            break

    # simple check for valid sprocket size. We require it
    # to be less than a third of the total scan height.
    # Otherwise, we give up and start the inner search
    # just from the center of the frame. This could be
    # improved - usually, the sprocket size stays pretty constant
    if (outerHigh - outerLow) < 0.3 * dy:
        searchCenter = (outerHigh + outerLow) // 2
    else:
        searchCenter = dy // 2

    # searching sprocket borders from the inside of the sprocket.
    # For this, the above found potential center of the sprocket
    # is used as a starting point to search for the sprocket edges
    innerLow = searchCenter
    for y in range(searchCenter, outerLow, -1):
        if smoothedHisto[y] > innerThreshold:
            innerLow = y
            break

    innerHigh = searchCenter
    for y in range(searchCenter, outerHigh):
        if smoothedHisto[y] > innerThreshold:
            innerHigh = y
            break

    # a simple sanity check again. We make sure that the
    # sprocket is larger than maxSize and smaller than
    # the outer boundaries detected. If so, not correction
    # is applied to the image
    sprocketSize = innerHigh - innerLow
    minSprocketSize = int(minSize * dy)

    if minSprocketSize < sprocketSize and sprocketSize < (outerHigh - outerLow):
        sprocketCenter = (innerHigh + innerLow) // 2
    else:
        sprocketCenter = dy // 2
        sprocketSize = 0

    # now try to find the sprocket edge on the right side
    # if requested. Only if a sprocket is detected at that point
    # Not optimized, quick hack...
    xShift = 0
    if horizontal and sprocketSize > 0:
        # calculate the region-of-interest

        # we start from the left edge of our previous roi
        # and look two times the
        rx0 = x0
        rx1 = x0 + 2 * (x1 - x0)

        # we use only a part of the whole sprocket height
        ry = int(0.8 * sprocketSize)
        ry0 = sprocketCenter - ry // 2
        ry1 = sprocketCenter + ry // 2

        # cutting out the roi
        horizontalStrip = img[ry0:ry1, rx0:rx1, :]

        # edge detection
        horizontalEdges = np.absolute(cv2.Sobel(horizontalStrip, cv2.CV_64F, 1, 0, ksize=3))

        # evidence accumulation
        histoHori = np.mean(horizontalEdges, axis=(0, 2))
        smoothedHori = cv2.GaussianBlur(histoHori, (1, 5), 0)

        # normalizing things
        maxPeakValueH = smoothedHori.max()
        thresholdHori = thresholds[1] * maxPeakValueH

        # now searching for the border
        xShift = 0
        for x in range((x1 - x0) // 2, len(smoothedHori)):
            if smoothedHori[x] > thresholdHori:
                xShift = x
                break

        # readjust calculated shift
        xShift = x1 - xShift

    return (xShift, dy // 2 - sprocketCenter)


## simple image transformation
def shiftImg(img, xShift, yShift):
    # create transformation matrix
    M = np.float32([[1, 0, xShift], [0, 1, yShift]])

    # ... and warp the image accordingly
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


# Multiprocess module - adjust the list of files passed to the function
def adjustFile(fileList):
    for f in fileList:
        inFile = None

        try:
            # Start the timer...
            tick = time.time()

            inFile = f[0]
            outFile = f[1]

            inputImg = cv2.imread(inFile)
            # run sprocket detection routine
            shiftX, shiftY = detectSprocketPos(inputImg, horizontal=True)
            # shift the image into its place
            outputImage = shiftImg(inputImg, shiftX, shiftY)
            # now output some test information
            dy, dx, dz = outputImage.shape

            # PIL treats and expects images in the normal, RGB order. But OpenCV expects and requires them to be in order of BGR so need to swap them
            imageRGB = cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB)

            # Convert numpy array to Pillow Iimage Object
            outputImage = Image.fromarray(imageRGB)
            outputImage.convert('RGB').save(outFile, format='JPEG', subsampling=0, quality=p.jpeg_quality)

            # Stop the timer...
            tock = time.time()

            print(
                f'Sprocket Detection in {(tock - tick) * 1000:.2f} msec, image size {dx} x {dy} - Applied shifts {shiftX}, {shiftY} - Output to -> {outFile}')

        except:
            print(f'{screen.colour.RED}Failed to process {inFile}{screen.colour.END}')


# Run main() when launched as a script
if __name__ == '__main__':
    main()


