# To run:   source setup.sh

arch_name="$(uname -m)"
if [ "${arch_name}" = "arm64" ]; then
    export CFLAGS=-I/$(brew --prefix)/include
    export CXXFLAGS=-I/$(brew --prefix)/include
    echo Setting Compiler flags for ARM wxPython build
else
    echo Not ARM - no compiler flags needed
fi

sleep 5


python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

