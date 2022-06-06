
Notes:

This program will process .mov and .avi files (haven't tested an AVI file) into a Jpeg stream, center each jpeg files based on 
sprocket position and recreate a movie from the jpeg stream.  Ffmpeg is used to extract and recreate the mov.  Ffmpeg is run externally as a
sub-process rather then using the python wrapper as the wrapper appears fairly slow in comparison.

- You cannot process the same folder twice - already processed folders will be skipped
- You can add new customer folders into a folder with already processed customer folders - only the new ones will be processed
- You can update an already processed customer folder to, for example, add another scanned reel only the newly added files will be processed
- Program includes a list of crazy folders which will be ignored.   Not so much of a problem for this script as it only iterates one folder deep.

- Files in the selected root folder will NOT be processed
- Files in the root of the customer folder will be processed
- Customer sub-folders will be ignored

** DO NOT used dots/periods/fullstops in filenames except in the suffix.   file.mov is fine, file.1.mov is NOT !
** Underscores, dashes and spaces etc. in names and folders are all fine.

