cd $HOME
git clone git@github.com:ggerganov/llama.cpp.git
cd $HOME/llama.cpp
cmake -B build
cmake --build build --config Release

cd $FAIR_CO2/workloads/llama
docker build --network=host --build-arg home=$HOME -t llama .

mv $FAIR_CO2/workloads/llama/model/*.gguf $HOME/llama.cpp/models/