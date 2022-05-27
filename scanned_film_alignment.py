import wx
import os
import platform
from multiprocessing import Process, cpu_count
import screen
import subprocess
import shlex
import sys
from PIL import Image
import time
import numpy as np
import cv2
import properties as p
import screen


class Disk:
    rootFolder              = None
    customerSubFolderList   = list()
    customerSubFolderCount  = 0
    startTime               = 0
    startTimeText           = None
    rawFilmScans            = list()
    rawJpegFolders          = list()
    adjustedJpegFolders     = list()
    adjustmentSrcDestList   = list()
    jpegFileCountToAdjust   = 0
    rawMovieFileCount       = 0


# GLobal instance to hold file and folder info
# Note: Every folder or file reference in here is full path
disk = Disk()




# Return the command line to run ffmpeg to split a movie into jpeg files
def ffmpeg_split_command( srcFile, dstFolder ): return f'ffmpeg -y -i "{srcFile}" -c:v copy "{dstFolder}frame_%d.jpg"'

# Return the command line to run ffmpeg to join jpeg files into a mov file
def ffmpeg_join_command( adjusted_folder, adjusted_movie ): return f'ffmpeg -y -framerate 25 -i "{adjusted_folder}frame_%d.jpg" -codec copy "{adjusted_movie}"'



# Popup dialog to select a destination folder
def popDialog_folder():
    app = wx.App()
    dialog = wx.DirDialog(None,'Choose Folder to process:',style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON )
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None

    dialog.Destroy()

    if path is None:
        print ('Conversion Cancelled')
        exit()

    return os.path.join(path, '')


# Get a list of folders off the root folder - this is the list of customers
# We need to process these one by one until complete
def getSubFolders(rootFolder):
    subFolders = next(os.walk(rootFolder))[1]
    return subFolders


def check_ffmpeg_installed(self):
    # Make sure ffmpeg is installed
    stream   = os.popen('which ffmpeg')
    ffmpeg_location = stream.read().strip()

    if not ffmpeg_location:
        return False
    else:
        return ffmpeg_location


# Note:  If folders already exist, we'll just exit.  
# Note:  I doubt anyone ever wants to run this on the same folder twice without cleaning up first
# Note:  and it will stop accidental screw-ups.

# Create a folder called folderName
# FolderName should be a full path unless you have already changed into a new working directory
# Note this will not create folders recursively - one at a time...
def createFolder(folderName):
    try:

        # Run through os.path.join in case it is missing the tailing /
        new_folder = os.path.join(folderName, '')

        # print (f'Debug: Would create ---->   {new_folder}')
        os.mkdir(folderName)
        print (f'Create Folder       : {new_folder}')

    except FileExistsError:
        print (f'Folder Create       : {new_folder} already exists')
        exit()
        
    except OSError as error:
        print (f'{screen.colour.RED}Unable to create folder {new_folder} {screen.colour.END}')
        print (f'{screen.colour.RED}Probably no available space left on this volume{screen.colour.END}')
        print (f'{screen.colour.RED}Stopping - something is broken{screen.colour.END}')
        print ( )
        exit()

    except:
        print (f'{screen.colour.RED}Unable to create folder {new_folder} {screen.colour.END}')
        print (f'{screen.colour.RED}Stopping - something is broken{screen.colour.END}')
        print ( )
        exit()


    return new_folder


# Run the ffmpeg session in a subprocess and monitor stdOut
# Runs subprocess with Popen/poll to show live stdOut
def run_ffmpeg(command, shellType=False, stdoutType=subprocess.PIPE):
    try:
        process = subprocess.Popen(
            shlex.split(command), shell=shellType, stdout=stdoutType)
    except:
        print (f'Error {sys.exc_info()[1]} while running {command} ')
        exit()

    while True:
        output = process.stdout.readline()

        if process.poll() is not None:
            break
        if output:
            print(output.strip().decode())

    rc = process.poll()
    return True

# Define the maximum allowed cores that this script can use
# Read entry from properties.py
def max_allowed_core_count():
    try:
        if p.thread_max < 1:                # Anything less than 0 means unrestricted so use cpu_max -1
            return (cpu_count() - 1)
        elif p.thread_max > cpu_count():    # Properties file is higher than number of cores in machine, so use cpu_max -1 
            return (cpu_count() - 1)
        else:
            return (p.thread_max)           # Configured value is OK
    except:
        print (f'Error - thread_max is not defined in properties file !  Will use maxim allowed CPU Cores for this job')
        return (cpu_count() -1 )            # Probably not set in properties files, so set to cpu_max -1

## simple sprocket detection algorithm
## returns the detected sprocket position
## relative to vertical scan center and left horizontal edge
def detectSprocketPos(img, roi = [0.01,0.09,0.2,0.8],       # region-of-interest - set as small as possible
                           thresholds = [0.5,0.2],          # edge thresholds; first one higher, second one lower
                           filterSize = 25,                 # smoothing kernel - leave it untouched
                           minSize = 0.1,                   # min. relative sprocket size to be expected - used for checks
                           horizontal = p.Horizontal):             # if you want a simple horizontal alignment as well

    ## inital preparations
    
    # get the size of the image     
    dy,dx,dz = img.shape

    # convert roi coords to real image coords
    x0 = int(roi[0]*dx)
    x1 = int(roi[1]*dx)
    
    y0 = int(roi[2]*dy)
    y1 = int(roi[3]*dy)        

    # cutting out the strip to work with
    # we're using the full scan heigth to simplify computations
    sprocketStrip = img[:,x0:x1,:]

    # now calculating the vertical sobel edges
    sprocketEdges = np.absolute(cv2.Sobel(sprocketStrip,cv2.CV_64F,0,1,ksize=3))

    # by averaging horizontally, only promient horizontal edges
    # show up in the histogram
    histogram     = np.mean(sprocketEdges,axis=(1,2))

    # smoothing the histogram to make signal more stable.
    # sigma==0 -> it is autocalculated
    smoothedHisto = cv2.GaussianBlur(histogram,(1,filterSize),0)

    # now analyzing the smoothed histogram
    # everything is relative to the detected maximum of the histogram
    # we only work in the region where the sprocket is expected
    maxPeakValue   = smoothedHisto[y0:y1].max()

    # the outer threshold is used to search for high peaks from the outside
    # it should be as high as possible in order to suppress that the algorithm
    # locks onto bright imprints of the film stock
    outerThreshold = thresholds[0]*maxPeakValue

    # the inner threshold is used to really search for the boundaries of
    # the sprocket. Implicitly it is assumed here that the area within
    # the sprocket is very evenly lit. If a lot of dust is present in
    # the material, this threshold should be raised higher
    innerThreshold = thresholds[1]*maxPeakValue

    # searching for the sprocket from the outside, first from below
    # we start not right at the border of the histogram in order to
    # avoid locking at bad cuts which look like tiny sprockets
    # to the algorithm    
    outerLow       = y0
    for y in range(y0,y1):
        if smoothedHisto[y]>outerThreshold:
            outerLow = y                 
            break
        
    # now searching from above
    outerHigh      = y1
    for y in range(y1,outerLow,-1):
        if smoothedHisto[y]>outerThreshold:
            outerHigh = y
            break

    # simple check for valid sprocket size. We require it
    # to be less than a third of the total scan height.
    # Otherwise, we give up and start the inner search
    # just from the center of the frame. This could be
    # improved - usually, the sprocket size stays pretty constant
    if (outerHigh-outerLow)<0.3*dy:
        searchCenter = (outerHigh+outerLow)//2
    else:
        searchCenter = dy//2

    # searching sprocket borders from the inside of the sprocket.
    # For this, the above found potential center of the sprocket
    # is used as a starting point to search for the sprocket edges 
    innerLow = searchCenter
    for y in range(searchCenter,outerLow,-1):
        if smoothedHisto[y]>innerThreshold:
            innerLow = y
            break
            
    innerHigh = searchCenter
    for y in range(searchCenter,outerHigh):
        if smoothedHisto[y]>innerThreshold:
            innerHigh = y
            break

    # a simple sanity check again. We make sure that the
    # sprocket is larger than maxSize and smaller than
    # the outer boundaries detected. If so, not correction
    # is applied to the image
    sprocketSize    = innerHigh-innerLow
    minSprocketSize = int(minSize*dy)
    if minSprocketSize<sprocketSize and sprocketSize<(outerHigh-outerLow) :
        sprocketCenter = (innerHigh+innerLow)//2
    else:
        sprocketCenter = dy//2
        sprocketSize   = 0
        
    # now try to find the sprocket edge on the right side
    # if requested. Only if a sprocket is detected at that point
    # Not optimized, quick hack...
    xShift = 0
    if horizontal and sprocketSize>0:
        # calculate the region-of-interest
        
        # we start from the left edge of our previous roi
        # and look two times the
        rx0 = x0
        rx1 = x0 + 2*(x1-x0)

        # we use only a part of the whole sprocket height
        ry = int(0.8*sprocketSize)
        ry0 = sprocketCenter-ry//2
        ry1 = sprocketCenter+ry//2

        # cutting out the roi
        horizontalStrip = img[ry0:ry1,rx0:rx1,:]

        # edge detection
        horizontalEdges = np.absolute(cv2.Sobel(horizontalStrip,cv2.CV_64F,1,0,ksize=3))

        # evidence accumulation
        histoHori       = np.mean(horizontalEdges,axis=(0,2))
        smoothedHori    = cv2.GaussianBlur(histoHori,(1,5),0)

        # normalizing things
        maxPeakValueH   = smoothedHori.max()
        thresholdHori   = thresholds[1]*maxPeakValueH
        
        # now searching for the border
        xShift = 0
        for x in range((x1-x0)//2,len(smoothedHori)):
            if smoothedHori[x]>thresholdHori:
                xShift = x                 
                break
            
        # readjust calculated shift
        xShift = x1 - xShift
    
    return (xShift,dy//2-sprocketCenter)


## simple image transformation
def shiftImg(img,xShift,yShift):
    
    # create transformation matrix
    M = np.float32([[1,0,xShift],[0,1,yShift]])

    # ... and warp the image accordingly
    return cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))



# Multiprocess module - adjust the list of files passed to the function 
def adjustFile(fileList):

    for f in fileList:
        try:
            # Start the timer...
            tick = time.time()

            inFile  = f[0]
            outFile = f[1]


            inputImg = cv2.imread(inFile)
            # run sprocket detection routine
            shiftX, shiftY = detectSprocketPos(inputImg, horizontal=True)
            # shift the image into its place
            outputImage = shiftImg(inputImg,shiftX,shiftY)
            # now output some test information
            dy,dx,dz = outputImage.shape


            # PIL treats and expects images in the normal, RGB order. But OpenCV expects and requires them to be in order of BGR so need to swap them
            imageRGB = cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB)

            # Convert numpy array to Pillow Iimage Object
            outputImage = Image.fromarray(imageRGB)
            outputImage.convert('RGB').save(outFile, format='JPEG', subsampling=0, quality=95)

            # Stop the timer...
            tock = time.time()

            print(f'Sprocket-Detection in {(tock-tick)*1000:.2f} msec, image size {dx} x {dy} - Applied shifts {shiftX}, {shiftY} - Output to -> {outFile}')

        except:
            print(f'{p.colour.RED}Failed to process {inFile}{p.colour.END}')



# Check ffmpeg is installed on the local system and available in the users $PATH
def check_ffmpeg_installed():
    # Make sure ffmpeg is installed
    stream   = os.popen('which ffmpeg')
    ffmpeg_location = stream.read().strip()

    if not ffmpeg_location:
        return False
    else:
        return ffmpeg_location



# Disk and file structure as follows:
# 
# /root/                            (top level folder holding customer folders)
# /root/customer                    (customer folder below root)
# /root/customer/file.avi           (raw film scans in the customer folder - could be more than one)
# /root/customer/file_raw           (folder to hold jpegs from raw movie scans - folder is same name as movie with no suffix and _raw)
# /root/customer/file_adjusted      (folder to hold adjusted jpegs created by ffmpeg - folder is same name as raw movie with no suffix and _adjusted)
# /root/customer/file_adjusted.mov  (adjusted movie files - movie is stored in customer folder with same name as original with a _adjusted inseryed before suffix


# Main process - all the work is done in here
# Could be broken out, but bit easier to follow this way
def main():


    print (screen.logo)
    print()

    global disk

    print(f'MacOS {platform.mac_ver()[0]}\n{cpu_count()} CPU Cores\nUsing {max_allowed_core_count()} CPU Cores \nffmpeg location ({check_ffmpeg_installed()})\n\n')
    
    # Get the Root Folder via Dialog Box
    disk.rootFolder = popDialog_folder()

    # Start the timer once the root folder is selected...
    disk.startTime = time.time()
    disk.startTimeText = time.strftime("%I:%M:%S")


    # Get a list of all the subFolders in the Root Folder
    disk.customerSubFolderList  = getSubFolders(disk.rootFolder)
    disk.customerSubFolderList  = [os.path.join(disk.rootFolder, folder, '') for folder in disk.customerSubFolderList ]
    disk.customerSubFolderList.sort()

    print (f'Root folder is {disk.rootFolder}')
    # print (f'Debug: ')
    # print (f'Debug: All Customer Folders')
    # for custFolder in disk.customerSubFolderList:
    #     print (f'Debug: {custFolder}')
    # print (f'Debug: ')


    # Count the subFolders and exit() if NONE
    disk.customerSubFolderCount = len(disk.customerSubFolderList)
    if disk.customerSubFolderCount  == 0:
        print (f'Found NO sub-folders to process - try another folder\n\n')
        exit(1)

    print (f'Found {disk.customerSubFolderCount} folders to process\n\n')

    # Get a list of all files in each customer folder
    # Remove anything that is not AVI or MOV
    # Create a list that has the full path to each raw Scan
    for customerFolder in disk.customerSubFolderList:

        # Find all files in the folder and filter out anything not avi or mov
        allFiles = next(os.walk(customerFolder))[2]
        filesofInterest = list()
        for file in allFiles:
            if file.lower().endswith('.mov') or file.lower().endswith('.avi'):
                fullPathtoFile = os.path.join(customerFolder, file)
                filesofInterest.append(fullPathtoFile)

        # if we found nothing of interest then skip this folder, otherwise add whatever we found to the central list
        if len(filesofInterest) == 0:
            print (f'Skipping Folder     : Skipping {customerFolder} -> No Movie Files found')
            continue
        else:
            disk.rawFilmScans += filesofInterest


        # print (f'Debug: ')
        # print (f'Debug: Found these Original Movie Files in {customerFolder}')
        # print (f'Debug: ')
        # for movie in filesofInterest:
        #     print (f'Debug: {movie}')
        # print (f'Debug: ')


    # Check how many raw movie files we found in all customer folders - if zero exit()
    disk.rawMovieFileCount = len(disk.rawFilmScans)
    if disk.rawMovieFileCount ==0:
        print (f'File Count          : Found NO Scans to adjust.  Will exit()')
        exit()
    else:
        print (f'File Count          : Found {disk.rawMovieFileCount} raw film scans to adjust.')


    print (f'Ready to Go.....')

    # print (f'Debug: ')
    # print (f'Debug: Full List of raw Movie Scans')
    # for m in disk.rawFilmScans:
    #     print (f'Debug: {m}')
    # print (f'Debug: ')


    # print (f'Debug : ')
    # print (f'Debug : ')
    # string = input('Press Enter')
    # print (f'Debug : ')



    # Create the folders to hold the raw and adjusted jpeg files for later processing
    for rawFile in disk.rawFilmScans:
        print ()
        # print (f'Debug: Creating Folders for {rawFile}')
        rawJpegFolder       = rawFile.split('.')[0] + '_raw'
        adjustedJpegFolder  = rawFile.split('.')[0] + '_adjusted'

        new_raw_folder      = createFolder(rawJpegFolder)
        new_adjusted_folder = createFolder(adjustedJpegFolder)

        disk.adjustedJpegFolders.append(new_adjusted_folder)
        disk.rawJpegFolders.append(new_raw_folder)

    # print (f'Debug: ')
    # print (f'Debug: List of all raw and adjusted jpeg folders for all discovered files')
    # for raw, adj in zip(disk.rawJpegFolders, disk.adjustedJpegFolders):
    #     print (f'Debug: Raw Folder        : {raw}')
    #     print (f'Debug: Adujusted Folder  : {adj}')

    # print (f'Debug: ')
        
    # print (f'Debug : ')
    # print (f'Debug : ')
    # string = input('Press Enter')
    # print (f'Debug : ')


    # Use ffmpeg to process each movie in disk.rawFilmScans and extract to its subfolder
    for rawMovie, rawFolder in zip(disk.rawFilmScans, disk.rawJpegFolders):
        print ()
        print (f'Processing Movie    : {rawMovie}')

        cli = ffmpeg_split_command (rawMovie, rawFolder)

        print (f'ffmpeg CLI          : {cli}')
        print ()

        # Run the command in a shell (bit nasty, but hey..)
        run_ffmpeg(cli)


    # Process the raw Jpeg files in each raw folder and make a [src, dst] list for the sprocket adjuster
    for rawFolder, adjustedFolder in zip(disk.rawJpegFolders, disk.adjustedJpegFolders):
        # Get all the files in the folder
        # Remove anything we didn't put there from our list of files to process

        # print (f'Debug: ')
        # print (f'Debug: processing Raw Folder {rawFolder} to build SRC, DST list for adjustment processing')
        # print (f'Debug: ')

        all_files       = next(os.walk(rawFolder))[2]
        all_raw_files   = [f for f in all_files if f.lower().endswith('.jpg')]

        # Create disk.adjustmentSrcDestList which is a list of all the raw Jpeg files created in all folders along with the file to write out after adjustment

        # List of Jpeg Files in each rawFolder
        for rawFile in all_raw_files:
            # print (f'Debug: raw file -> {rawFile}')

            # Generate a clean full path for rawFile and adjusted file
            # Drop into a list and append to the fill list of src/dst files to be processes
            rFile = os.path.join(rawFolder, rawFile)
            aFile = os.path.join(adjustedFolder, rawFile)
            disk.adjustmentSrcDestList.append([rFile, aFile])

            # print (f'Debug: Adding file -> {[rFile, aFile]} to disk.adjustmentSrcDestList')

    # Sort the list of adjustments to keep in some sort of order
    disk.adjustmentSrcDestList.sort()

    # Count how many files we have to adjust - quit if no files found
    disk.jpegFileCountToAdjust = len(disk.adjustmentSrcDestList)
    if disk.jpegFileCountToAdjust == 0 :
        print (f'File Count          : Found NO Files to adjust.  Will exit()')
        exit()
    else:
        print (f'File Count          : Found {disk.jpegFileCountToAdjust:,d} Files to adjust')     #Note - {int:,d} shows comma seperators for numbers 


    # print (f'Debug : ')
    # print (f'Debug : ')
    # string = input('Press Enter')
    # print (f'Debug : ')


    # Set core count to 1 if we have less files than cores.  Likely slower to start multiple sub-processes
    if len(disk.adjustmentSrcDestList) < max_allowed_core_count():
        core_count = 1
    else:
        core_count = max_allowed_core_count()


    print ()
    print ()
    print (f'Multiprocessing     : Using {core_count} CPU Cores')
    print ()
    

    # print (f'Debug : ')
    # print (f'Debug : ')
    # string = input('Press Enter')
    # print (f'Debug : ')


    # Split the full list of files into one for each CPU Core to process
    # Returns a list of lists
    perProcessFileLists = np.array_split(disk.adjustmentSrcDestList, core_count)


    # Create a list of sub processes
    subProcesses = list()
    for fileList in perProcessFileLists:
        subProcesses.append(Process(target=adjustFile, args=(fileList, )))
        
    for proc in subProcesses:
        proc.start()

    for proc in subProcesses:
        proc.join()

    print ()
    print ('All files adjusted!')
    print ()
    print (f'Starting to combine {disk.jpegFileCountToAdjust:,d} file into {disk.rawMovieFileCount} adjusted movies' )


    # print (f'Debug : ')
    # print (f'Debug : ')
    # string = input('Press Enter')
    # print (f'Debug : ')


    # Sort the list of movies and adjustedJpegFolders
    # Probably unnecessary, but just in case anything has gotten out of sync 
    disk.adjustedJpegFolders.sort()
    disk.rawFilmScans.sort()


    # Process the adjusted Jpeg folders into New Movies using ffmpeg
    for adjustedFolder, rawMovie in zip(disk.adjustedJpegFolders, disk.rawFilmScans):
        new_movie_name = rawMovie.split('.')[0] + '_adjusted.mov'
        print () 
        print ()
        print (f'Adjusted Folder               : {adjustedFolder}')
        print (f'Original Movie Name           : {rawMovie}')
        print (f'New Movie Name                : {new_movie_name}')

        cli = ffmpeg_join_command (adjustedFolder, new_movie_name)

        print (f'ffmpeg CLI          : {cli}')
        print ()

        # Run the command in a shell (bit nasty, but hey..)
        run_ffmpeg(cli)

    # print (f'Debug : ')
    # print (f'Debug : ')
    # string = input('Press Enter')
    # print (f'Debug : ')


    print ()
    print ()
    print ('ALL DONE')
    print ()
    print ()
    print ()
    summaryDisplay()



# Show a little summary at the end
def summaryDisplay():
    print ( )
    print ( f'{screen.colour.YELLOW}- Summary -------------{screen.colour.END}')
    print ( )
    print ( f'Total Adjusted Files  : {disk.jpegFileCountToAdjust:,d}' )
    print ( f'Total Adjusted Movies : {disk.rawMovieFileCount:,d}' )
    print ( f'Start time            : {disk.startTimeText}' )
    print ( f'End time              : {time.strftime("%I:%M:%S")}' )
    print ( f'Time to convert       : {time.time() - disk.startTime:.2f} seconds' )
    print ()
    print ()



# Run main() when launched as a script
if __name__ == '__main__':
    main()


