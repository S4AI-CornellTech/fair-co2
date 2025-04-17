cd $HOME
git clone --recursive https://github.com/intel/pcm.git
sudo apt-get update
sudo apt install -y cmake
cd pcm
mkdir build
cd build
sudo cmake ..
sudo cmake --build . --parallel
sudo cmake --install .
sudo modprobe msr
cd $FAIR_CO2