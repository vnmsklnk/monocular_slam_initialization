sudo apt-get install -y libsuitesparse-dev libeigen3-dev
cd monocular_slam_initialization/g2opy/g2opy
git apply ../g2opy.patch
mkdir build
cd build
cmake ..
make -j8
cd ..
python3 setup.py install --user
