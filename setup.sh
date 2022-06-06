# To run:   source setup.sh

# Needed for wxPython on arm64 based Macs
#arch_name="$(uname -m)"
#if [ "${arch_name}" = "arm64" ]; then
#    export CFLAGS=-I/$(brew --prefix)/include
#    export CXXFLAGS=-I/$(brew --prefix)/include
#    echo Setting Compiler flags for ARM wxPython build
#else
#    echo Not ARM - no compiler flags needed
#fi


#Try to install python-tk via homebrew if not already installed
tk_installed="$(brew info python-tk)"

if [[ "$tk_installed" == *"Not installed"* ]]; then
   brew install python-tk@3.9
else
   echo python-tk already brewed - moving right along...
fi



python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

