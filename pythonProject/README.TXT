The key package is autoML auto-sklearn. For now, it doesn't install great with python 10, 11, or 12.
It will also not install with versions of numpy > 14.5
It also has to be installed on Linux.

I ended up being able to install on Linux on Mac0s M2 using ARM64 architecture. A bit of a small miracle.
Here is what you need to do:

For M chips only:
Install a VM, I used UTM mostly bc VMWare had an error when I tried to log in and get Fusion 13 Player.
I chose to virtualize on ARM64. Was not sure if that was smart, but figured it'd run faster if I'm not trying to emulate anything.
Go through the whole setup.
Pro tip is when you're done setting up and the VM is rebooting, it needs to not reboot from disc so "eject" the image using the top right button
or else you just get a blank screen with a blinking horizontal line.

Install Linux desktop just bc it makes finding other downloads easier.

Install pycharm - download the tar file and follow the commands below
Don't worry, it works. I was wondering if it was actually AMD64 arch but it's ok.
chmod +x pycharm-community-2024.1-aarch64.tar.gz
sudo tar -xzf pycharm-community-2024.1-aarch64.tar.gz
./pycharm.sh

Install miniforge3. This one is tricky bc you do need to use the specific Linux aarch64 executable.
It took me a while to fine, but I'm giving you the link here.
Run the executable with bash.
It will come with python 10.x but don't worry, we'll make a different environment
Create and environment: conda create -n <env_nam> python=3.9
Make sure you put the python=3.9 on the end of it, it tells the environment to install 3.9
Now do NOT install numpy or scikit.

Do this: sudo apt-get install gcc python3-dev
and then
Straight away install auto-sklearn.
pip install auto-sklearn.
The latest package installs the numpy tools it needs. If you already have numpy installed, start over bc it's not worth trying to
make sure all pieces uninstalled if you used pip.

To start Pycharm from the terminal, navigate to pycharm folder (most likely in Downloads)
then  run the executable at bin/pycharm.sh
Go to "tools" and select "Create Desktop Entry" so that you'll have a shortcut on the desktop toolbar


--- also needed conda install h5py and pip install tensorflow
