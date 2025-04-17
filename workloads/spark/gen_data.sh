# Download and setup tpcds-kit
cd $HOME
sudo apt-get install -y gcc make flex bison byacc git g++-9 gcc-9
git clone git@github.com:gregrahn/tpcds-kit.git
cd tpcds-kit/tools
make CC=gcc-9 OS=LINUX

# Generate dataset
mkdir $FAIR_CO2/workloads/spark/data
cd $HOME/tpcds-kit/tools
./dsdgen -dir $FAIR_CO2/workloads/spark/data -delimiter , -scale 100 -force Y -suffix .csv -table store_sales