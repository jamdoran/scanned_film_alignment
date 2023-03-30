
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


-------------------------------------------------------------------------------------------
Setup process on a clean Mac
-------------------------------------------------------------------------------------------

Install iTerm for sanity sake

# Install xcode command line tools 
xcode-select --install
    Wait for this to finish - it can take a while

# Accept the xcode license 
sudo xcodebuild -license accept


# Install HomeBrew as per instructions on home page
# Add brew to the path
(echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> /Users/jamdoran/.zprofile


#Install Python 10, zsh etc - this will take a while 
brew install python@3.10 zsh python-tk@3.10 ffmpeg
chsh -s /usr/local/bin/zsh

#Open a new iTerm tab and check we have the correct zsh
$ which zsh
/usr/local/bin/zsh   //correct


# Clone the repos into the Git Folder
mkdir ~/Documents/Git
cd ~/Documents/Git
git clone https://github.com/jamdoran/scanned_film_alignment/
git clone https://github.com/jamdoran/tiff_bulk_convert


# Add alias to .zshrc
alias convert='cd ~/Documents/Git/tiff_bulk_convert && source .venv/bin/activate && python tiff_convert.py'
alias sprocket='cd ~/Documents/Git/scanned_film_alignment && source .venv/bin/activate && python scanned_film_alignment.py'

#Should be good to go 

