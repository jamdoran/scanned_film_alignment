# To run:   source setup.sh

# Install python-tk version 3.10 specifically
brew install python-tk@3.10

# make sure to use python3.10
python3.10 -m venv .venv
source .venv/bin/activate

#Upgrade and install everything necessary
python -m pip install --upgrade pip
python -m pip install -r requirements.txt





#Setup process on a clean Mac


#Install iTerm for sanity sake

# Install xcode command line tools
#xcode-select --install
#    Wait for this to finish - it can take a while


# Install HomeBrew as per instructions on home page
# Add brew to the path
#(echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> /Users/jamdoran/.zprofile


#Install Python 10, zsh etc
#brew install python@3.10 zsh python-tk@3.10 ffmpeg
#chsh -s /usr/local/bin/zsh

#Open a new iTerm tab and check we have the correct zsh
#$ which zsh
#/usr/local/bin/zsh   //correct


# Clone the repos into the Git Folder
#mkdir ~/Documents/Git
#cd ~/Documents/Git
#git clone https://github.com/jamdoran/scanned_film_alignment/
#git clone https://github.com/jamdoran/tiff_bulk_convert


# Add alias to .zshrc
#alias convert='cd ~/Documents/Git/tiff_bulk_convert && source .venv/bin/activate && python tiff_convert.py'
#alias sprocket='cd ~/Documents/Git/scanned_film_alignment && source .venv/bin/activate && python scanned_film_alignment.py'

#Should be good to go

