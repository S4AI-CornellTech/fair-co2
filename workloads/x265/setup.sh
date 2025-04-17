# Get x265 video
sudo apt-get install -y p7zip-full
cd $HOME
wget http://ultravideo.cs.tut.fi/video/Bosphorus_3840x2160_120fps_420_8bit_YUV_Y4M.7z
7z x Bosphorus_3840x2160_120fps_420_8bit_YUV_Y4M.7z
mkdir -p $FAIR_CO2/workloads/x265/data
mv $HOME/Bosphorus_3840x2160.y4m $FAIR_CO2/workloads/x265/data/Bosphorus_3840x2160.y4m
rm -rf $HOME/Bosphorus_3840x2160_120fps_420_8bit_YUV_Y4M.7z

cd $FAIR_CO2/workloads/x265
docker build --network=host -t x265 .
cd $FAIR_CO2


